# LLM provider routing

Authoritative Agent calls use server-owned roles. The request body may carry a
legacy `llm_provider` value for compatibility, but once
`llm_role_bindings.json` exists it is ignored for role-routed calls.

The supported roles are:

| Call path | Role |
| --- | --- |
| Planner, Execution Agent, Replanner, and scientific Agent session control-plane calls | `control_plane` |
| `map_oled_contextual_semantics` | `scientific_mapping` |

The role file lives beside the private `llm_profiles.json` under
`MOLLY_CONFIG_DIR` (or the configured user config directory):

```json
{
  "schema_version": "llm_role_bindings.v1",
  "bindings": {
    "control_plane": "provider-a",
    "scientific_mapping": "deepseek"
  }
}
```

Named profiles are server-configured in the private `profiles` object of
`llm_profiles.json`. Secrets still use the existing environment, keyring, or
file secret-source rules; they are not stored in the profile document.

Each profile declares its structured-output and role capabilities explicitly:

```json
{
  "profiles": {
    "provider-a": {
      "profile_id": "provider-a",
      "provider": "openai_compatible",
      "endpoint": "https://control.example.test/v1",
      "model": "control-model",
      "api_key_source": "environment",
      "api_key_env": "MOLLY_CONTROL_PLANE_API_KEY",
      "capabilities": {
        "structured_output_mode": "native_json_schema",
        "control_plane_eligible": true,
        "scientific_mapping_eligible": true
      }
    },
    "deepseek": {
      "profile_id": "deepseek",
      "provider": "openai_compatible",
      "endpoint": "https://api.deepseek.com/v1",
      "model": "deepseek-model",
      "api_key_source": "environment",
      "api_key_env": "MOLLY_SCIENTIFIC_MAPPING_API_KEY",
      "capabilities": {
        "structured_output_mode": "json_object_local_validation",
        "control_plane_eligible": false,
        "scientific_mapping_eligible": true
      }
    }
  }
}
```

`structured_output_mode` is a transport contract, not provider-name logic:
`native_json_schema` requests provider-side JSON Schema enforcement, while
`json_object_local_validation` requests a JSON object and relies on the
existing local Pydantic/JSON Schema validation. A profile that is not
`control_plane_eligible` cannot back authoritative orchestration.

Control-plane acceptance is intentionally a hard gate: replay the same full
Planner payload 10 times and the same full Execution Agent payload 20 times;
every response must pass the complete structured contract. Any schema drift
fails the 100% gate for authoritative use, regardless of the mapping role's
eligibility.
