# Security notes

- Do not commit `.env`, API keys, local credentials, or private model endpoints.
- Keep provider credentials in environment variables referenced by `api_key_env`; do not put
  raw keys in `openfusion.yaml`.
- Use `OPENFUSION_API_KEY` when exposing the server outside localhost.
- The default bind host is `127.0.0.1`. Only use `0.0.0.0` with a strong
  `OPENFUSION_API_KEY` and network controls.
- Prefer local providers for confidential prompts.
- Add network-level access control if running in production.
- Review provider data-retention policies before sending sensitive data to cloud models.
- Logs should avoid storing full prompts by default in a production deployment.
