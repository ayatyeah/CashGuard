FROM golang:1.25-alpine AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -o /out/server ./cmd/server
RUN CGO_ENABLED=0 go build -trimpath -o /out/migrate ./cmd/migrate

FROM alpine:3.21
RUN apk add --no-cache ca-certificates && addgroup -S app && adduser -S -G app app && mkdir -p /data/uploads && chown -R app:app /data
WORKDIR /app
COPY --from=build /out/server /out/migrate /app/
USER app
ENV ADDR=0.0.0.0:8080 DATA_DIR=/data GIN_MODE=release
EXPOSE 8080
CMD ["/app/server"]
