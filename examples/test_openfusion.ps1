$token = $env:OPENFUSION_API_KEY
if ([string]::IsNullOrWhiteSpace($token)) {
  $token = "replace-with-a-long-random-token"
}

$headers = @{
  "Authorization" = "Bearer $token"
  "Content-Type"  = "application/json"
}

$body = @{
  model = "openfusion/critique-revision"
  messages = @(
    @{
      role = "user"
      content = "Give a short plan for building a RAG chatbot."
    }
  )
  max_tokens = 256
  fusion_max_total_calls = 6
} | ConvertTo-Json -Depth 10

$response = Invoke-RestMethod `
  -Uri "http://localhost:8000/v1/chat/completions" `
  -Method Post `
  -Headers $headers `
  -Body $body

$response | ConvertTo-Json -Depth 20
