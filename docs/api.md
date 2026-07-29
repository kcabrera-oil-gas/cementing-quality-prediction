# API Reference

## Authentication

Use `X-API-Key` header.

## GET /

Returns service info and available models.

## POST /predict

Run inference. Body: `{"features": {"feature": value}}`
