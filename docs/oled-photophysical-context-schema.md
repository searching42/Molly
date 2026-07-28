# OLED Photophysical Context Schema

This contract adds three human-reviewed molecule/interaction properties without
making measurements from incompatible experimental contexts appear directly
comparable.

## Canonical properties

| Property id | Canonical unit | Allowed layers | Accepted aliases |
| --- | --- | --- | --- |
| `photoluminescence_peak_nm` | `nm` | molecule, interaction | `PL maximum`, `PL peak`, `PL_max` |
| `prompt_lifetime_ns` | `ns` | interaction | `prompt PL lifetime`, `prompt fluorescence lifetime` |
| `delayed_lifetime_us` | `us` | interaction | `delayed PL lifetime`, `delayed fluorescence lifetime` |

The broad aliases `emission peak`, `prompt lifetime`, and `delayed lifetime` are
intentionally excluded because they do not identify a sufficiently precise
property without additional interpretation.

## Comparison context

Every property uses the same explicit comparison-context fields:

- `measurement_temperature` and `measurement_temperature_unit`
- `host_material`
- `dopant_concentration` and `dopant_concentration_unit`
- `sample_form`
- `excitation_wavelength` and `excitation_wavelength_unit`
- `lifetime_fit_method`

Missing values remain `null`; they are never inferred. Numeric temperature,
dopant concentration, and excitation wavelength values are normalized while the
source condition remains available on the original observation.

Canonical observations carry one of three context states:

- `not_required`: the property does not use this context contract;
- `incomplete`: one or more required fields are missing;
- `complete`: all required fields are present and a stable comparison-context
  hash is available.

`oled_observations_are_directly_comparable(...)` returns true for these
photophysical properties only when both observations have complete context,
the same property and normalized unit, and the same context hash.

## Dataset behavior

Incomplete observations remain valid evidence-bearing schema records and emit
an `incomplete_photophysical_comparison_context` warning. They cannot enter a
comparable curated intrinsic view. Complete observations carry their context
hash and context features into the view; different contexts therefore remain
separate during deduplication.

This contract does not infer missing experimental details, resolve molecular
identity, admit device-only records, or write gold/training data.
