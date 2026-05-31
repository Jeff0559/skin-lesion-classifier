# NLP Approach Comparison: A vs B

## Test Setup
- 20 labelled skin lesion descriptions
- Gold labels from HAM10000 classes: mel, nv, bcc, akiec, bkl, df, vasc

## Results

| Approach | Model | Accuracy | Latency | Cost |
|---|---|---|---|---|
| A | Sentence-Transformer (all-MiniLM-L6-v2) | 70% (14/20) | 55 ms/sample | $0.00 |
| B | GPT-4o-mini (OpenAI API) | 75% (15/20) | 664 ms/sample | ~$0.001/20 calls |

**Winner: B (GPT-4o-mini)**

## Per-Case Results

| Description | Gold | Approach A | A✓ | Approach B | B✓ |
|---|---|---|---|---|---|
| dark irregular spot growing for 6 months | mel | mel | ✓ | mel | ✓ |
| small pearly nodule on face | bcc | df | ✗ | bcc | ✓ |
| itchy scaly red patch on arm | akiec | akiec | ✓ | akiec | ✓ |
| brown mole, stable for years | nv | nv | ✓ | nv | ✓ |
| rough warty lesion on back | bkl | df | ✗ | bkl | ✓ |
| small firm bump on leg | df | df | ✓ | bkl | ✗ |
| red vascular spot, bleeds easily | vasc | vasc | ✓ | vasc | ✓ |
| asymmetric dark lesion with irregular border | mel | mel | ✓ | mel | ✓ |
| flesh-colored raised bump | bcc | df | ✗ | bkl | ✗ |
| flat brown spot, sun-damaged skin | bkl | akiec | ✗ | akiec | ✗ |
| multiple small moles, unchanged | nv | nv | ✓ | nv | ✓ |
| crusty lesion on scalp | akiec | akiec | ✓ | bcc | ✗ |
| bluish-black nodule | mel | nv | ✗ | mel | ✓ |
| spider-like red vessels on skin | vasc | vasc | ✓ | vasc | ✓ |
| soft movable lump under skin | df | df | ✓ | bkl | ✗ |
| pink scaly patch, recurring | akiec | akiec | ✓ | akiec | ✓ |
| dark spot with satellite lesions | mel | mel | ✓ | mel | ✓ |
| waxy stuck-on appearance | bkl | bkl | ✓ | bkl | ✓ |
| translucent bump with blood vessels | bcc | vasc | ✗ | bcc | ✓ |
| uniform brown oval mole | nv | nv | ✓ | nv | ✓ |

## Analysis

- **Approach A** (Sentence-Transformer): Fast (local, no API), free, 70% accuracy.
  Weakness: Template-based class prototypes limit discrimination for visually similar classes (df vs bkl).
- **Approach B** (GPT-4o-mini): Higher accuracy (75%), better at understanding natural language nuance.
  Weakness: API latency (~664 ms/sample), requires API key, small cost per call.
- **Recommendation**: Use Approach B for production (better accuracy). Use Approach A as fast fallback.
