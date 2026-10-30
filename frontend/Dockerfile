# Build the static app, then serve it with nginx, which also forwards the API routes to the
# backend so the browser sees one origin (the same job Vite's proxy does in development).
FROM node:24.22.0-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY . .
RUN npm run build

FROM nginx:1.29.3-alpine
# Rendered with envsubst at start (the image's templates feature): BACKEND_URL, FRONTEND_PORT.
COPY nginx.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html
ENV BACKEND_URL=http://127.0.0.1:8000 \
    FRONTEND_PORT=8080
