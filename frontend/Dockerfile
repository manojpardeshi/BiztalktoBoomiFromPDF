# Frontend Dockerfile
FROM node:20-alpine AS base
WORKDIR /app

# Install deps
COPY package*.json ./
RUN npm ci

# Copy sources and build
COPY . .
RUN npm run build

# Serve with a lightweight static server
FROM nginx:alpine AS runner
# Copy Vite dist to Nginx html dir
COPY --from=base /app/dist /usr/share/nginx/html

# Copy a basic nginx.conf to handle SPA routes
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
