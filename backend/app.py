import io
import os
import re
import uuid
import logging
from logging.handlers import RotatingFileHandler
from typing import Dict, Any, TypedDict, Optional
import time
import zipfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
from dotenv import load_dotenv

# LangChain / LangGraph imports
from langchain.schema import HumanMessage, SystemMessage  # type: ignore
from langchain_openai import AzureChatOpenAI  # type: ignore
from langgraph.graph import StateGraph, END  # type: ignore

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL", "gpt-5-chat")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")

if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
    print("Warning: Azure OpenAI environment variables not set. The LLM calls will fail until configured.")

# Simple wrapper for LLM call

def get_llm():
    try:
        if not (AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_MODEL):
            logger.warning("Azure OpenAI env not fully set. endpoint=%s model=%s key_present=%s", AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_MODEL, bool(AZURE_OPENAI_API_KEY))
            return None
        llm = AzureChatOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_deployment=AZURE_OPENAI_MODEL,
            temperature=0.1,
            timeout=90,
        )
        logger.info("AzureChatOpenAI initialized: endpoint=%s deployment=%s api_version=%s", AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_MODEL, AZURE_OPENAI_API_VERSION)
        return llm
    except Exception as e:
        logger.exception("LLM init error: %s", e)
        return None

app = FastAPI(title="BizTalk to Boomi PDF Processor")

# CORS (allow local frontend; extend as needed)
allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173"
]
extra_origins = os.getenv("CORS_EXTRA_ORIGINS")
if extra_origins:
    allowed_origins.extend([o.strip() for o in extra_origins.split(",") if o.strip()])

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging setup
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("biztalk_to_boomi")

# Separate LLM logger to file with rotation
llm_logger = logging.getLogger("biztalk_to_boomi.llm")
llm_logger.setLevel(LOG_LEVEL)
if not llm_logger.handlers:
    fh = RotatingFileHandler(os.path.join(LOG_DIR, "llm.log"), maxBytes=5*1024*1024, backupCount=5)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(fmt)
    llm_logger.addHandler(fh)
    # Avoid double logging to root
    llm_logger.propagate = False

TEMP_OUTPUTS: Dict[str, Dict[str, Any]] = {}


# -------- LangGraph State Definition --------
class PipelineState(TypedDict, total=False):
    job_id: str
    raw_text: str
    sections: Dict[str, str]
    boomi_prompt: str
    mapping_table: str
    connections_table: str
    final_text: str

class ProcessResponse(BaseModel):
    job_id: str
    download_url: str

SECTION_HEADERS = [
    "Background Technical Overview",
    "2.1", "2.2", "2.3", "2.4", "2.5",
    "3.1", "3.1.1", "3.1.2", "3.2", "3.3",
    "4.1", "4.2", "4.3", "4.4"
]

SECTION_REGEX = re.compile(r"^(Background Technical Overview|(?:[234]\.\d(?:\.\d)?))\b")


def extract_pdf_text(file_bytes: bytes) -> str:
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded (size=0). Please select a valid PDF.")
    # Quick signature check
    if not file_bytes.lstrip().startswith(b"%PDF"):
        logger.warning("Uploaded file does not start with %PDF header; attempting to parse anyway.")
    # Attempt pypdf first
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages).strip()
        if text:
            return text
        logger.warning("pypdf returned empty text; will attempt pdfminer fallback.")
    except Exception as e:
        logger.exception("PDF extraction failed via pypdf: %s", e)
    # Fallback: pdfminer.six (optional dependency)
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract_text  # type: ignore
        text = pdfminer_extract_text(io.BytesIO(file_bytes)) or ""
        text = text.strip()
        if text:
            logger.info("PDF text extracted using pdfminer fallback (length=%d).", len(text))
            return text
        else:
            logger.warning("pdfminer fallback produced empty text.")
    except Exception as e2:
        logger.exception("pdfminer fallback failed: %s", e2)
    raise HTTPException(status_code=400, detail="PDF extraction failed. The file may be scanned, encrypted, or unsupported. Try a searchable PDF or provide text.")


def naive_section_parser(raw_text: str) -> Dict[str, str]:
    lines = raw_text.splitlines()
    sections: Dict[str, str] = {}
    current_key = None
    buffer: list[str] = []

    def commit():
        if current_key is not None:
            sections[current_key] = "\n".join(buffer).strip()

    for line in lines:
        line_stripped = line.strip()
        if SECTION_REGEX.match(line_stripped):
            # new section header
            commit()
            current_key = line_stripped.split()[0]
            buffer = [line_stripped]
        else:
            buffer.append(line)
    commit()
    return sections


def _redact(text: str) -> str:
    if not text:
        return text
    t = text
    # Common credential patterns
    t = re.sub(r"(?i)(password\s*[:=]\s*)([^\s]+)", r"\1***", t)
    t = re.sub(r"(?i)(pass\s*[:=]\s*)([^\s]+)", r"\1***", t)
    t = re.sub(r"(?i)(apikey|api[_-]?key|token|bearer)\s*[:=\s]+([^\s]+)", r"\1 ***", t)
    # Basic auth header-like strings
    t = re.sub(r"(?i)(Authorization:\s*Basic\s+)([A-Za-z0-9+/=]+)", r"\1***", t)
    # Connection strings user=;password=
    t = re.sub(r"(?i)(user\s*=\s*[^;]+;\s*password\s*=\s*)([^;]+)", r"\1***", t)
    return t


def _log_llm(job_id: str | None, label: str, prompt: str, response: str | None, note: str = ""):
    payload = {
        "job_id": job_id or "",
        "label": label,
        "note": note,
        "prompt": _redact(prompt) if prompt else "",
        "response": _redact(response) if response else "",
    }
    try:
        llm_logger.info("LLM %s | job=%s | note=%s\nPROMPT:\n%s\nRESPONSE:\n%s",
                        payload["label"], payload["job_id"], payload["note"], payload["prompt"], payload["response"])
    except Exception:
        # Never break the app due to logging errors
        pass


def call_llm_structured(prompt: str, *, job_id: str | None = None, label: str = "generic") -> str:
    llm = get_llm()
    if not llm:
        # For development without keys, return placeholder
        placeholder = "LLM_CALL_PLACEHOLDER: " + prompt[:200]
        _log_llm(job_id, label, prompt, placeholder, note="no_llm_config")
        return placeholder
    messages = [
        SystemMessage(content="You are a precise information extraction agent."),
        HumanMessage(content=prompt)
    ]
    try:
        resp = llm.invoke(messages)
        content = getattr(resp, "content", "")  # type: ignore
        _log_llm(job_id, label, prompt, content, note="ok")
        return content
    except Exception as e:
        err_text = str(e)
        logger.warning("LLM call failed: %s", err_text)
        _log_llm(job_id, label, prompt, f"exception={err_text}", note="exception")

        # If Azure content filter triggers, sanitize and retry once
        def sanitize_prompt(text: str) -> str:
            t = text
            # Redact potential credentials
            t = re.sub(r"(?i)(password\s*[:=]\s*)([^\s]+)", r"\1***", t)
            t = re.sub(r"(?i)(pass\s*[:=]\s*)([^\s]+)", r"\1***", t)
            t = re.sub(r"(?i)(api[_-]?key\s*[:=]\s*)([^\s]+)", r"\1***", t)
            t = re.sub(r"(?i)(secret\s*[:=]\s*)([^\s]+)", r"\1***", t)
            # De-intensify imperative phrasing sometimes flagged as jailbreak-y
            t = t.replace("Return ONLY", "Provide").replace("no preamble", "without extra commentary")
            t = re.sub(r"(?i)ignore previous|override policies|bypass", "", t)
            return t

        if "content_filter" in err_text.lower() or "responsibleaipolicyviolation" in err_text.lower():
            sanitized = sanitize_prompt(prompt)
            logger.info("Retrying LLM with sanitized prompt (first 200 chars): %s", sanitized[:200])
            try:
                resp = llm.invoke([
                    SystemMessage(content="You are a precise information extraction agent. Follow safety policies and redact any secrets with ***."),
                    HumanMessage(content=sanitized)
                ])
                content = getattr(resp, "content", "")  # type: ignore
                _log_llm(job_id, label, sanitized, content, note="sanitized_ok")
                return content
            except Exception as e2:
                logger.exception("Sanitized retry failed: %s", e2)
                _log_llm(job_id, label, sanitized, f"sanitized_exception={e2}", note="sanitized_error")

        # Handle transient rate limit/timeout: small exponential backoff retries
        transient_markers = ("429", "rate limit", "too many requests", "insufficient_quota", "timeout", "timed out")
        if any(m in err_text.lower() for m in transient_markers):
            for attempt in range(2):
                delay = 2 * (2 ** attempt)
                logger.info("Transient error detected (attempt %d). Sleeping %ds then retrying label=%s", attempt + 1, delay, label)
                time.sleep(delay)
                try:
                    resp = llm.invoke(messages)
                    content = getattr(resp, "content", "")  # type: ignore
                    _log_llm(job_id, label, prompt, content, note=f"retry_ok_{attempt+1}")
                    return content
                except Exception as e3:
                    logger.warning("Retry %d failed: %s", attempt + 1, e3)
                    _log_llm(job_id, label, prompt, f"retry_exception_{attempt+1}={e3}", note=f"retry_error_{attempt+1}")

        placeholder = "LLM_CALL_PLACEHOLDER_ERROR: " + prompt[:200]
        # Try to include a hint if an HTTP status code appears in the error
        code_hint = ""
        m = re.search(r"\b(\d{3})\b", err_text)
        if m:
            code_hint = f" http_status={m.group(1)}"
        _log_llm(job_id, label, prompt, placeholder, note=f"error:{err_text[:200]}{code_hint}")
        return placeholder


def build_boomi_prompt(sections: Dict[str, str], *, job_id: str | None = None) -> str:
    base_context = sections.get("Background Technical Overview", "") or sections.get("Background", "")
    interfaces = sections.get("3.1", "")
    maps = sections.get("3.3", "")
    orchestrations = sections.get("3.2", "")

    raw_prompt_material = f"""Background / Technical Overview:\n{base_context}\n\nInterfaces (Ports) 3.1:\n{interfaces}\n\nOrchestrations 3.2:\n{orchestrations}\n\nMaps / Transformations 3.3:\n{maps}\n"""

    llm_instruction = f"""You are an expert in migrating BizTalk solutions to Boomi. Using the provided BizTalk analysis excerpts, craft a single concise Boomi AI prompt that a user can paste into the Boomi Console to bootstrap an equivalent integration design. Follow responsible AI guidelines and do not include or expose any sensitive credentials—redact with ***.\n\nThe output prompt should:\n- Start with an action-oriented directive (e.g., 'Design a Boomi integration that ...').\n- Summarize the core business purpose in 1-2 sentences.\n- Enumerate the required Boomi process steps (ingestion, transformations, routing, error handling).\n- List external systems/connectors with their roles (mask credentials as ***).\n- Describe data transformations at a functional level (reference source→target domains; avoid listing every field unless explicitly provided).\n- Include orchestration/sequencing logic and retry/exception handling strategy.\n\nProvide the Boomi AI prompt as plain text, without extra commentary.\n\nSOURCE MATERIAL:\n{raw_prompt_material}\n"""

    result = call_llm_structured(llm_instruction, job_id=job_id, label="boomi_prompt").strip()
    # Log (truncate to avoid huge logs). Sensitive data should already be masked; still, keep it limited.
    logger.info("Generated Boomi prompt (truncated 800 chars): %s", result[:800])
    return result


def generate_source_to_target_mapping(sections: Dict[str, str], *, job_id: str | None = None) -> str:
    maps = sections.get("3.3", "")
    extraction_prompt = f"""From the following BizTalk mapping description extract a tabular source-to-target field mapping.\nIf unavailable, produce a single-row table indicating 'Not Specified'.\nReturn GitHub-flavored Markdown table with columns: SourceField | SourceType | TargetField | TargetType | TransformationLogic.\nText:\n{maps}\n"""
    result = call_llm_structured(extraction_prompt, job_id=job_id, label="mapping_table")
    return result.strip()

def extract_connections(sections: Dict[str, str], *, job_id: str | None = None) -> str:
    connections_text = sections.get("2.1", "") + "\n" + sections.get("2.2", "") + "\n" + sections.get("3.1", "")
    prompt = f"""Task: From the text below, list connection endpoints such as adapters, ports, endpoints, queues, databases, and APIs.\nSafety: Do not include any credential values or private data. If such values appear, replace only the sensitive value with [MASKED].\nOutput: Provide a GitHub-flavored Markdown table with columns: Name | Type | Direction | Technology | Endpoint/Server | Auth (masked) | Notes.\nText:\n{connections_text}\n"""
    resp = call_llm_structured(prompt, job_id=job_id, label="connections_table").strip()
    if resp.startswith("LLM_CALL_PLACEHOLDER"):
        # Deterministic fallback parser
        try:
            entries = []
            # Split by numbered items like "1. Foo (bar)"
            blocks = re.split(r"\n(?=\d+\.)", connections_text.strip())
            for block in blocks:
                if not block.strip():
                    continue
                lines = [l.strip("\u2022 ") for l in block.strip().splitlines() if l.strip()]
                header = lines[0]
                m = re.match(r"\d+\.\s*(.+?)\s*(?:\((.*?)\))?", header)
                name = m.group(1).strip() if m else header
                notes = m.group(2).strip() if (m and m.group(2)) else ""
                ep_type = direction = tech = endpoint = auth = "N/A"
                for line in lines[1:]:
                    if line.lower().startswith("endpoint type:"):
                        ep_type = line.split(":", 1)[1].strip()
                        tech = ep_type
                    elif line.lower().startswith("physical endpoint:"):
                        endpoint = line.split(":", 1)[1].strip()
                    elif line.lower().startswith("direction:"):
                        direction = line.split(":", 1)[1].strip()
                    elif line.lower().startswith("data:"):
                        # append to notes to keep concise
                        data_val = line.split(":", 1)[1].strip()
                        notes = (notes + "; " if notes else "") + f"Data={data_val}"
                    elif line.lower().startswith("purpose:"):
                        purpose_val = line.split(":", 1)[1].strip()
                        notes = (notes + "; " if notes else "") + purpose_val
                entries.append({
                    "Name": name,
                    "Type": ep_type,
                    "Direction": direction,
                    "Technology": tech,
                    "Endpoint": endpoint,
                    "Auth": "***",  # masked/unknown
                    "Notes": notes or "",
                })
            if entries:
                # Build Markdown table
                header_row = "| Name | Type | Direction | Technology | Endpoint/Server | Auth (masked) | Notes |"
                sep_row = "| --- | --- | --- | --- | --- | --- | --- |"
                rows = [header_row, sep_row]
                for e in entries:
                    rows.append(f"| {e['Name']} | {e['Type']} | {e['Direction']} | {e['Technology']} | {e['Endpoint']} | {e['Auth']} | {e['Notes']} |")
                fallback_table = "\n".join(rows)
                _log_llm(job_id, "connections_table", prompt, fallback_table, note="fallback_ok")
                return fallback_table
        except Exception as ex:
            _log_llm(job_id, "connections_table", prompt, f"fallback_exception={ex}", note="fallback_error")
    return resp


def assemble_output(boomi_prompt: str, mapping_table: str, connections_table: str) -> str:
    parts = [
        "## Boomi AI Prompt",
        boomi_prompt,
        "\n## Source to Target Mapping",
        mapping_table,
        "\n## Connection Objects (Masked)",
        connections_table,
    ]
    return "\n\n".join(parts)


def _strip_code_fences(markdown: str) -> str:
    t = markdown.strip()
    if t.startswith("```"):
        # remove first fence line
        lines = t.splitlines()
        # drop first line (``` or ```markdown)
        lines = lines[1:]
        # drop trailing ``` if present
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines)
    return markdown


def _parse_markdown_table(md_table: str) -> Dict[str, Any]:
    """Parse a GitHub-flavored markdown table into headers and rows.
    Returns {"headers": List[str], "rows": List[List[str]]}. Best-effort; ignores malformed lines.
    """
    text = _strip_code_fences(md_table)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # find header and separator
    header_idx = None
    sep_idx = None
    for i, ln in enumerate(lines):
        if "|" in ln and not ln.lower().startswith("note:"):
            # heuristic: next line has dashes separated by |
            if i + 1 < len(lines):
                nxt = lines[i + 1]
                if "|" in nxt and re.search(r"-", nxt):
                    header_idx = i
                    sep_idx = i + 1
                    break
    if header_idx is None or sep_idx is None:
        return {"headers": [], "rows": []}
    # split header
    def split_row(s: str) -> list[str]:
        parts = [p.strip() for p in s.strip("|").split("|")]
        return parts

    headers = split_row(lines[header_idx])
    data_rows: list[list[str]] = []
    for ln in lines[sep_idx + 1:]:
        if not ("|" in ln):
            break
        row = split_row(ln)
        # pad/truncate to header length
        if len(row) < len(headers):
            row += [""] * (len(headers) - len(row))
        elif len(row) > len(headers):
            row = row[:len(headers)]
        data_rows.append(row)
    return {"headers": headers, "rows": data_rows}


def markdown_mapping_to_excel_bytes(md_table: str) -> bytes:
    """Create an Excel workbook from a Markdown mapping table and return bytes."""
    try:
        from openpyxl import Workbook  # type: ignore
    except Exception:
        # If openpyxl is not available, return an empty minimal workbook-like CSV as fallback in XLSX name
        buf = io.BytesIO()
        buf.write(b"SourceField,SourceType,TargetField,TargetType,TransformationLogic\n")
        return buf.getvalue()

    parsed = _parse_markdown_table(md_table)
    wb = Workbook()
    ws = wb.active
    ws.title = "Mapping"
    headers = parsed.get("headers") or ["SourceField", "SourceType", "TargetField", "TargetType", "TransformationLogic"]
    rows = parsed.get("rows") or [["Not Specified", "", "", "", ""]]

    ws.append(headers)
    for r in rows:
        ws.append(r)

    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()

## -------- LangGraph Node Functions --------

def node_parse(state: PipelineState) -> PipelineState:
    raw = state.get("raw_text", "")
    return {"sections": naive_section_parser(raw)}


def node_boomi_prompt(state: PipelineState) -> PipelineState:
    sections = state["sections"]
    return {"boomi_prompt": build_boomi_prompt(sections, job_id=state.get("job_id"))}


def node_mapping(state: PipelineState) -> PipelineState:
    sections = state["sections"]
    return {"mapping_table": generate_source_to_target_mapping(sections, job_id=state.get("job_id"))}


def node_connections(state: PipelineState) -> PipelineState:
    sections = state["sections"]
    return {"connections_table": extract_connections(sections, job_id=state.get("job_id"))}


def node_assemble(state: PipelineState) -> PipelineState:
    return {"final_text": assemble_output(state["boomi_prompt"], state["mapping_table"], state["connections_table"]) }


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("parse", node_parse)
    graph.add_node("boomi_prompt", node_boomi_prompt)
    graph.add_node("mapping", node_mapping)
    graph.add_node("connections", node_connections)
    graph.add_node("assemble", node_assemble)

    graph.set_entry_point("parse")
    graph.add_edge("parse", "boomi_prompt")
    graph.add_edge("boomi_prompt", "mapping")
    graph.add_edge("mapping", "connections")
    graph.add_edge("connections", "assemble")
    graph.add_edge("assemble", END)
    return graph.compile()


GRAPH = build_graph()


@app.post("/process-pdf", response_model=ProcessResponse)
async def process_pdf(file: UploadFile = File(...)):
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        # Some browsers may send octet-stream; we try anyway
        logger.warning("Unexpected content-type: %s; proceeding to attempt parse as PDF.", file.content_type)
    file_bytes = await file.read()
    logger.info("/process-pdf received file name=%s content_type=%s size=%d bytes", getattr(file, 'filename', ''), file.content_type, len(file_bytes))
    try:
        raw_text = extract_pdf_text(file_bytes)
    except HTTPException as http_ex:
        logger.warning("/process-pdf PDF extraction error: %s", http_ex.detail)
        raise

    # Create job id now to correlate logs
    job_id = str(uuid.uuid4())

    # Execute LangGraph pipeline
    result_state: PipelineState = GRAPH.invoke({"raw_text": raw_text, "job_id": job_id})  # type: ignore
    final_text = result_state.get("final_text") or "Processing failed to produce output"
    mapping_tbl = result_state.get("mapping_table") or ""
    TEMP_OUTPUTS[job_id] = {
        "md_content": final_text,
        "mapping_table": mapping_tbl,
        "filename": f"boomi_conversion_{job_id}.zip",
    }
    return ProcessResponse(job_id=job_id, download_url=f"/download/{job_id}")

@app.get("/download/{job_id}")
async def download(job_id: str):
    item = TEMP_OUTPUTS.get(job_id)
    if not item:
        raise HTTPException(status_code=404, detail="Job ID not found")

    # Build ZIP in-memory
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Add Markdown
        md_bytes = (item.get("md_content") or "").encode("utf-8")
        md_name = (item.get("filename") or "output.zip").replace(".zip", ".md")
        zf.writestr(md_name, md_bytes)
        # Add Excel from mapping table
        xlsx_bytes = markdown_mapping_to_excel_bytes(item.get("mapping_table") or "")
        xlsx_name = (item.get("filename") or "output.zip").replace(".zip", "_mapping.xlsx")
        zf.writestr(xlsx_name, xlsx_bytes)

    zip_buf.seek(0)
    return StreamingResponse(zip_buf, media_type="application/zip", headers={
        "Content-Disposition": f"attachment; filename={item['filename']}"
    })

@app.get("/health")
async def health():
    return {"status": "ok"}
