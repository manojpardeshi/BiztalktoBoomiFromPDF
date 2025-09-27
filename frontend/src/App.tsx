import React, { useState } from 'react';

interface ApiResponse {
  job_id: string;
  download_url: string;
}

const App: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [downloadLink, setDownloadLink] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;
    setError(null);
    setLoading(true);
    setDownloadLink(null);
    try {
      const formData = new FormData();
      formData.append('file', file);
      const resp = await fetch('http://localhost:8000/process-pdf', {
        method: 'POST',
        body: formData,
      });
      if (!resp.ok) {
        throw new Error(await resp.text());
      }
      const data: ApiResponse = await resp.json();
      setDownloadLink(`http://localhost:8000${data.download_url}`);
    } catch (err: any) {
      setError(err.message || 'Upload failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <div className="w-full max-w-xl bg-white shadow rounded p-8 space-y-6">
        <h1 className="text-2xl font-semibold text-gray-800">BizTalk to Boomi PDF Processor</h1>
        <form onSubmit={handleSubmit} className="space-y-4">
          <input
            type="file"
            accept="application/pdf"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
            className="block w-full text-sm text-gray-700 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100"
            required
          />
          <button
            type="submit"
            disabled={!file || loading}
            className="px-4 py-2 rounded bg-indigo-600 text-white disabled:opacity-50"
          >
            {loading ? 'Processing...' : 'Process PDF'}
          </button>
        </form>
        {error && <div className="text-red-600 text-sm">{error}</div>}
        {downloadLink && (
          <div className="mt-4">
            <a
              href={downloadLink}
              className="text-indigo-700 underline font-medium"
              download
            >
              Download Result
            </a>
          </div>
        )}
        <p className="text-xs text-gray-500">Ensure the PDF follows the expected BizTalk analysis structure.</p>
      </div>
    </div>
  );
};

export default App;
