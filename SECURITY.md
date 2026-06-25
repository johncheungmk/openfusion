# Security and privacy notes

## Secrets

- Never commit `.env`, API keys, bearer tokens, private endpoints, or cloud credentials.
- Reference provider credentials through `api_key_env`.
- OpenFusion rejects unknown fields in provider configuration, including an inline `api_key` field.
- Provider HTTP error snippets are bounded and known API keys are redacted.

## Network exposure

- The default server bind address is `127.0.0.1`.
- Set a strong `OPENFUSION_API_KEY` before binding to `0.0.0.0`.
- Use a reverse proxy, TLS, firewall rules, rate limits, and upstream budget controls for shared deployments.
- Provider `base_url` values are trusted administrator configuration. Do not allow untrusted users to edit them; doing so could create SSRF access to internal services.

## Multi-model data exposure

A fusion request may send the same prompt to several providers. Review every provider's retention and training policy before processing confidential data.

Use:

```yaml
include_candidate_outputs: false
include_workflow_outputs: false
```

when intermediate text should not be returned to clients. These settings do not prevent the configured providers from receiving the prompt.

## Planner safety

The optional model planner returns data, not executable code. OpenFusion validates that data against:

- a fixed strategy allowlist;
- currently enabled provider names;
- bounded sample and refinement counts;
- the remaining call budget.

The model planner cannot add arbitrary tools, shell commands, URLs, or Python functions.

## Prompt injection

Candidate models may produce instructions aimed at the critic, reviser, or synthesizer. Prompts label candidate text as untrusted evidence and instruct workflow roles not to treat it as authority. This reduces but does not eliminate prompt-injection risk. Use provider isolation and task-specific verification for high-stakes deployments.

## Chain-of-thought

OpenFusion prompts request concise user-visible conclusions and explicitly prohibit exposing hidden chain-of-thought. Workflow traces contain operational metadata rather than private reasoning.

## Denial of service and cost

- Set `fusion.max_total_calls`.
- Set provider timeouts and upstream token/rate budgets.
- Keep `max_parallel` appropriate for local hardware.
- Large local CPU models may occupy a machine for minutes per multi-stage request.
