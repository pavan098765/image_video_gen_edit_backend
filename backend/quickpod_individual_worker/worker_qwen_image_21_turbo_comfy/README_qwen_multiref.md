# Qwen Image 2.1 Turbo multi-reference workflow

Generic image-edit graph for Qwen Image 2.1 Turbo in ComfyUI. Reference images are inputs. The prompt for that run decides what each image means. The graph does not rewrite prompts and does not bake in a subject, industry, or photography style.

Prompt rewriting happens before this workflow. Follow `docs/system_prompt_edit.txt` upstream, then pass `rewritten_prompt` in as the prompt string. This graph does not load that guide and does not run a prompt-enhancement model.

## Files

| File | Role |
| --- | --- |
| `workflow/qwen_image_2_1_turbo_multiref_workflow.json` | Standard ComfyUI workflow. Open this in the UI. |
| `workflow/qwen_image_2_1_turbo_multiref_api.json` | Same graph in API prompt format for `POST /prompt`. |
| `workflow/qwenImage21Turbo8Steps_turbo8StepV10.json` | Original text-to-image workflow. Not modified. |
| `workflow/qwenImage21Turbo8Steps_turbo8StepV10_api.json` | Original API workflow. Not modified. |

The pod `/run` path loads `qwen_image_2_1_turbo_multiref_api.json`. A job with no images is text-to-image. A job with `init_image` (one URL or a list) or `image_1` … `image_10` is a reference edit. The worker adds `LoadImage` nodes only for the images in that request, so missing placeholder files are never queued.

On the pod, open `qwen_image_2_1_turbo_multiref_workflow.json` in ComfyUI. Queue it as it is for text-to-image. Unmute a prefix of reference loaders for an edit.

## What the graph keeps

- Diffusion model: `unet/qwen-image-2.1-turbo-Q8_0.gguf` through `UnetLoaderGGUF` (molbal ComfyUI-GGUF, already pinned in `nodes.lock.json`).
- Text encoder: `text_encoders/qwen3vl_8b_int8_convrot.safetensors` with CLIP type `qwen_image`.
- VAE: `vae/qwen_image_2.1_vae_bf16.safetensors`, connected to both the encoder and the decoder.
- Sampler: `euler`, `cfg` 1, `add_noise` true.
- Turbo schedule on `ManualSigmas`, unchanged: `1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0`.

No extra custom nodes are required. `TextEncodeQwenImage21` and `ComfySwitchNode` ship with ComfyUI 0.37.

`ResolutionSelector` uses a multiple of 32. The original text-to-image file used 8. Qwen Image 2.1 resizes references onto a 32-pixel grid, and the existing job patch already forces 32, so the edit graph matches that grid.

## How images reach the model

Each `reference_N` loader feeds `use_reference_N` (`ComfySwitchNode`, switch on). That switch feeds `TextEncodeQwenImage21` input `images.image_N`.

The node accepts up to **16** references (`image_1` … `image_16`). This workflow exposes **10**, the same count as the official Qwen Image 2.1 edit templates.

With the VAE connected, each supplied image is resized and encoded as a reference latent that is spliced into the sequence. The tokenizer also prepends a vision block labeled `<image1>`, `<image2>`, and so on, in the order of the images that are actually present. The workflow does not treat those tags as text substitutions. They have to match the images that were passed.

The encoder skips a slot that is null. Later slots then move forward. `<image3>` is the third image only when `reference_1`, `reference_2`, and `reference_3` are all supplied. Leave unused slots at the end.

The node reads the first frame of each image batch.

## Slot map

| Loader | Encoder input | Prompt tag | Default |
| --- | --- | --- | --- |
| `reference_1` | `images.image_1` | `<image1>` | Muted. `use_reference_1` is on. |
| `reference_2` | `images.image_2` | `<image2>` | Muted. `use_reference_2` is on. |
| `reference_3` … `reference_10` | `images.image_3` … `images.image_10` | `<image3>` … `<image10>` | Muted. Matching `use_reference` switch is on. |

Muted loaders are `mode` 2. ComfyUI leaves them out of the queued prompt, so a missing `reference_N.png` does not fail the queue. The links are already drawn.

To use the first K images:

1. Unmute `reference_1` through `reference_K` only. Do not unmute a later slot while an earlier one stays muted.
2. Set each unmuted loader to a file in `ComfyUI/input`.
3. Leave `use_reference_N` on. If that switch is off, the encoder receives nothing for that slot and later tags shift.

## Canvas size

`resolution` on the encoder is a pixel budget, not a width or a height. The default is `1024` (about 1 megapixel). `0` keeps each reference at its own size, snapped to a multiple of 32. `2048` is about the native 2K budget.

`custom_size` defaults to on, for both the UI graph and the API graph. Text-to-image then samples `EmptyLatentImage` from `ResolutionSelector`. No reference file is required.

For a reference edit, turn `custom_size` off so the sampler uses the encoder latent. That latent follows the **first supplied** reference after the resolution resize. This is a size rule, not a meaning assigned to slot 1. The prompt still defines every image's role. Leave `custom_size` on during an edit only when you want the ResolutionSelector size instead. A canvas far from the first reference can shift the edit.

The `/run` handler does this itself: no images keeps `custom_size` on; images turn it off unless the job sets `custom_size`, `aspect_ratio`, `wh_ratio`, or both `width` and `height`.

Upstream `system_prompt_edit.txt` can return `wh_ratio` or `ratio_follow`. This graph does not read that JSON.

- `wh_ratio`: set `ResolutionSelector` to that aspect and turn `custom_size` on.
- `ratio_follow` pointing at `<image1>`: leave `custom_size` off.
- `ratio_follow` pointing at a later image: do not reorder the files. Set width and height to that image's snapped size and turn `custom_size` on.

## Prompt

The prompt widget is empty. Paste the rewritten instruction.

At `cfg` 1 the negative prompt is unused. Leave it empty unless you raise `cfg` on purpose. The published Turbo path keeps `cfg` at 1.

For one reference, the upstream guide says not to put a tag in the rewritten prose. The tokenizer still attaches that image as the first vision block.

For two or more references, the rewritten prompt must name `<image1>`, `<image2>`, … in slot order, and say what is taken from each.

## API prompt

`qwen_image_2_1_turbo_multiref_api.json` is a text-to-image prompt. It has no `LoadImage` nodes, so ComfyUI can queue it without reference files. `custom_size` is on. `SamplerCustom.latent_image` comes from that switch, whose `on_true` input is `EmptyLatentImage`.

`/run` keeps that shape when the job has no images. When the job includes references, the worker saves them into `ComfyUI/input` and adds `images.image_1` … `images.image_N` on `TextEncodeQwenImage21`, up to 10. It does not rewrite the prompt.

Accepted reference fields, in order:

- `image_1` … `image_10`, or `reference_1` … `reference_10`. A missing number shifts the later tags.
- otherwise `init_image` as one URL or a list of URLs
- otherwise `images`, `reference_images`, `image_urls`, or `init_images`
- otherwise `first_frame_url`, `image_url`, or `image`

Each value can be an `http` URL or a base64 image. More than 10 images are truncated.

| Node | Input | Type | Default |
| --- | --- | --- | --- |
| `TextEncodeQwenImage21` | `prompt` | string | `""` |
| `TextEncodeQwenImage21` | `negative_prompt` | string | `""` |
| `TextEncodeQwenImage21` | `resolution` | int | `1024` |
| `TextEncodeQwenImage21` | `clip`, `vae` | CLIP, VAE links | loaders |
| `custom_size` | `switch` | boolean | `true` for text-to-image, `false` when references are attached and no explicit size was sent |
| `custom_size` | `on_false` | LATENT link | encoder latent |
| `custom_size` | `on_true` | LATENT link | `EmptyLatentImage` |
| `ResolutionSelector` | `aspect_ratio`, `megapixels`, `multiple` | combo, float, int | `1:1 (Square)`, `1`, `32` |
| `SamplerCustom` | `cfg`, `noise_seed`, `add_noise` | float, int, bool | `1`, `0`, `true` |

## Example prompts

These are caller prompts. They are not stored in the workflow.

One reference, local edit. The worker turns `custom_size` off unless you also send a size. In the UI, turn `custom_size` off yourself.

```text
Change the jacket to a light wool coat and keep the person, pose, background, and framing unchanged.
```

Two references. Image 1 is the scene to keep. Image 2 supplies a person.

```text
Keep the composition, lighting, and background of <image1>. Place the person from <image2> into that scene, preserving their face and clothing.
```

Two references. Image 1 is a product. Image 2 is a room.

```text
Place the product from <image1> on the table in <image2>. Match the room lighting and keep the rest of the room unchanged.
```

Three references. Image 1 is a person, image 2 is clothing, image 3 is a background.

```text
Keep the person's identity from <image1>, dress them in the outfit from <image2>, and replace the background with the scene in <image3>.
```

Three references of one person, used only as identity, with a new composition. Set `custom_size` on and choose the output aspect, because none of the photos is the canvas.

```text
Using the same person shown in <image1>, <image2>, and <image3>, create a new three-quarter portrait in a plain studio. Preserve the face across all three references.
```

## Limitations

- The node accepts 16 references. This graph and `/run` use 10.
- Text-to-image is the queued default: `custom_size` on, no reference files.
- A reference edit follows the first supplied image unless `custom_size`, `aspect_ratio`, `wh_ratio`, or both `width` and `height` are set.
- A later `ratio_follow` target is applied by sending that size. The graph does not reorder images.
- `cfg` 1 ignores the negative prompt.
- An explicit canvas far from the first reference can shift the edit.
- The UI workflow wires 10 reference slots. The API file does not include loaders until `/run` adds the ones the job sent.
- No image was generated here. ComfyUI is not running on this machine. The check below is structural.

## Structural check

Both JSON files parse. In the UI graph every link endpoint exists, `reference_N` reaches `images.image_N` for N = 1..10, those loaders are muted, and `custom_size` is on. The API graph has no `LoadImage` nodes. Its sampler latent comes from `custom_size`, which selects `EmptyLatentImage`. Patching that API graph with no images leaves it as text-to-image. Patching it with two filenames adds `images.image_1` and `images.image_2` and turns `custom_size` off. Turbo sigmas, euler, cfg 1, and the GGUF, CLIP, and VAE filenames match the original workflow. The prompt widget is empty.
