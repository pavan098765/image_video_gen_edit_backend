# Generate Music

## OpenAPI Specification

```yaml
openapi: 3.0.1
info:
  title: ''
  description: ''
  version: 1.0.0
paths:
  /api/v1/jobs/createTask:
    post:
      summary: Generate Music
      deprecated: false
      description: >
        :::warning Document updated

        If you have already completed the integration process previously, you
        can still access the old version document at: Old version address
        (https://docs.kie.ai/old-model/suno-api/generate-music)

        :::

        Generate music with or without lyrics using AI models.


        ### Usage Guide

        - This endpoint creates music based on your text prompt

        - Multiple variations will be generated for each request

        - You can control detail level with custom mode and instrumental
        settings


        ### Parameter Details


        * Always required: `custom_mode`, `instrumental`, `model`

        * In Custom Mode (`custom_mode: true`):
          * `title` is required, maximum 80 characters
          * `prompt` is optional; when provided, it is used strictly as lyrics and sung in the generated track. If `lyrics` is also provided, `lyrics` takes priority
          * At least one of `style`, `lyrics`, or `negative_tags` must be provided; generation is rejected if all are empty
          * If `instrumental: true`: generate instrumental music (no vocals)
          * If `instrumental: false`: use `lyrics` as lyrics (fall back to `prompt` if `lyrics` is not provided)
          * Optional: `duration`, `negative_tags`, `vocal_gender`, `style_weight`, `weirdness_constraint`, `audio_weight`, `variety`, `persona_id`, `persona_model`
          * Character limits by model:
            * ~~**V4**: `prompt` 3000 characters, `style` 200 characters~~
            * ~~**V4\_5, V4\_5PLUS, V4\_5ALL, V5, V5_5**: `prompt` 5000 characters, `style` 1000 characters~~
            * **V6, V6_MINI, V6_WILD**: `style` 1000 characters; `lyrics` 5000 characters
          * `title` length limit: 80 characters (all models)
        * In Non-custom Mode (`custom_mode: false`):
          * `prompt` is optional; it serves as the core idea, and lyrics are generated automatically (not a strict match), maximum 3000 characters
          * `lyrics` can be used as a lyrics attachment together with `prompt`
          * At least one of `image_urls`, `video_urls`, `audio_urls`, `style`, or `lyrics` must be provided; generation is rejected if all are empty
          * `image_urls`, `video_urls`, and `audio_urls` are only valid in this mode
          * Do not pass parameters that are only effective in custom mode: `duration`, `vocal_gender`, `style_weight`, `weirdness_constraint`, `audio_weight`, `variety`
          * Total attachments must not exceed 10: `style` + `lyrics` + `image_urls` + `video_urls` + `audio_urls`

        ### Optional Parameters


        * `prompt` (string): Description of the desired audio content. Optional.
        Used as lyrics in custom mode; used as the core idea in non-custom mode
        (maximum 3000 characters).

        * `lyrics` (string): Lyrics content. Optional. V6 maximum 5000
        characters. In custom mode, takes priority over `prompt` as lyrics; in
        non-custom mode, can be used as a lyrics attachment together with
        `prompt`.

        * `image_urls` (array): Image references. Only effective when
        `custom_mode` is `false`. Up to 5 images, each no more than 10 MB.
        Supported formats: jpeg, png, webp, bmp.

        * `video_urls` (array): Video references. Only effective when
        `custom_mode` is `false`. Up to 1 file, each no more than 100 MB,
        duration no more than 241 seconds. Supported formats: mp4, mov, webm.

        * `audio_urls` (array): Audio references. Only effective when
        `custom_mode` is `false`. Duration must be between 6 seconds and 30
        minutes; each file no more than 500 MB.

        * `style` (string): Music style specification. See character limits in
        Parameter Details above.

        * `negative_tags` (string): Music styles or traits to exclude from the
        generated audio. Optional.

        * `vocal_gender` (string): Vocal gender preference. `m` for male, `f`
        for female. Only effective when `custom_mode` is `true`. In practice,
        this only increases probability and cannot guarantee the instruction is
        followed.

        * `style_weight` (number): Strength of adherence to the specified style.
        Range 0–1, up to 2 decimal places. Only effective when `custom_mode` is
        `true`.

        * `weirdness_constraint` (number): Creative/experimental deviation.
        Range 0–1, up to 2 decimal places. Only effective when `custom_mode` is
        `true`.

        * `audio_weight` (number): Relative weight of audio features. Range 0–1,
        up to 2 decimal places. Only effective when `custom_mode` is `true`. Not
        supported when there are no vocals.

        * `variety` (number): Diversity of generated results. Integer from 0–4,
        default 1. Only effective when `custom_mode` is `true`. `0` off (exact
        style), `1` normal (default, balanced), `2` high (distinct styles), `3`
        extra (bold exploration), `4` max (maximum variation).

        * `persona_id` (string): Persona ID or Voice ID. Optional. To generate a
        Persona ID, see [Generate Persona](/suno-api/generate-persona).

        * `persona_model` (string): Persona model, `style_persona` or
        `voice_persona`. Only available for V5 (Discontinued), V5.5
        (Discontinued), V6, V6_MINI, and V6_WILD.

        * `duration` (number): Target length in seconds (10–360). Optional; if
        omitted, the model chooses the length. Only valid when `custom_mode` is
        `true` and the model is ~~V5_5~~, V6, V6_MINI, or V6_WILD. Do not pass
        it otherwise.


        ### Developer Notes

        - Recommendation for new users: Start with `custom_mode: false` for
        simpler usage

        - Generated files are retained for 14 days

        - Callback process has three stages: `text` (text generation), `first`
        (first track complete), `complete` (all tracks complete)
      operationId: generate-music
      tags:
        - docs/en/Market/Music Models/Suno/Music Generation
      parameters: []
      requestBody:
        content:
          application/json:
            schema:
              type: object
              required:
                - model
                - input
              properties:
                model:
                  type: string
                  description: The model name to use for generation. Required field.
                  examples:
                    - ai-music-api/generate
                callBackUrl:
                  type: string
                  format: uri
                  description: >-
                    The URL to receive generation task completion updates.
                    Optional but recommended for production use.


                    - System will POST task status and results to this URL when
                    generation completes

                    - Callback includes generated content URLs and task
                    information

                    - Your callback endpoint should accept POST requests with
                    JSON payload containing results

                    - Alternatively, poll GET /api/v1/jobs/recordInfo?taskId=...
                    (see [Get Task Details](/market/common/get-task-detail)).
                    For Suno audio tasks the tracks are in data.response.data[]
                    (audio_url, stream_audio_url, image_url, id, duration).

                    - To ensure callback security, see [Webhook Verification
                    Guide](/common-api/webhook-verification) for signature
                    verification implementation
                  examples:
                    - https://api.example.com/callback
                input:
                  type: object
                  description: Input parameters for the generation task
                  properties:
                    prompt:
                      type: string
                      description: >-
                        A description of the desired audio content. Optional.

                        - Custom mode (`custom_mode=true`): When provided, the
                        prompt is used strictly as lyrics and sung in the
                        generated track. Character limits by model:
                          - V4: Maximum 3000 characters
                          - V4_5, V4_5PLUS, V4_5ALL, V5_5, V5: Maximum 5000 characters
                        - Non-custom mode (`custom_mode=false`): When provided,
                        the prompt serves as the core idea. Lyrics are generated
                        automatically based on it and do not strictly match the
                        input. Maximum 3000 characters.

                        `prompt` is not required. In non-custom mode, provide at
                        least one of `image_urls`, `video_urls`, `audio_urls`,
                        `style`, or `lyrics`. In custom mode, provide at least
                        one of `style`, `lyrics`, or `negative_tags`. Generation
                        is rejected if all related inputs are empty.
                      examples:
                        - A calm and relaxing piano track with soft melodies
                    style:
                      type: string
                      description: >-
                        Music style specification for the generated audio.

                        - Character limits by model:
                          - **V4** (Discontinued): Maximum 200 characters
                          - **V4_5 and V4_5PLUS** (Discontinued): Maximum 1000 characters
                          - **V4_5ALL** (Discontinued): Maximum 1000 characters
                          - **V5_5 and V5** (Discontinued): Maximum 1000 characters
                          - **V6, V6_MINI, and V6_WILD**: Maximum 1000 characters
                        - Common examples: Jazz, Classical, Electronic, Pop,
                        Rock, Hip-hop, etc.
                      examples:
                        - Classical
                    title:
                      type: string
                      description: |-
                        Title for the generated music track.
                        - Required in Custom Mode (`custom_mode: true`).
                        - Max length: 80 characters.
                        - Will be displayed in player interfaces and filenames.
                      examples:
                        - Peaceful Piano Meditation
                    custom_mode:
                      type: boolean
                      description: >-
                        Determines whether advanced parameter customization is
                        enabled.

                        - If `true`: Allows detailed control

                        - If `false`: Simplified mode
                      examples:
                        - true
                    instrumental:
                      type: boolean
                      description: >-
                        Whether to generate instrumental music (no vocals). See
                        Parameter Details for the fields required in each mode.
                      examples:
                        - true
                    model:
                      type: string
                      description: >-
                        Required. Use V6, V6_MINI or V6_WILD.

                        - **`V6_WILD`**: Pushes creative boundaries for bolder,
                        more distinctive musical expression.

                        - **`V6_MINI`**: Lightweight and fast, balancing quality
                        and speed for effortless creation.

                        - **`V6`**: Greater musical expression with more natural
                        vocals and richer details.
                      enum:
                        - V4
                        - V4_5
                        - V4_5PLUS
                        - V4_5ALL
                        - V5
                        - V5_5
                        - V6
                        - V6_MINI
                        - V6_WILD
                      x-apidog-enum:
                        - value: V4
                          name: ''
                          description: ''
                        - value: V4_5
                          name: ''
                          description: ''
                        - value: V4_5PLUS
                          name: ''
                          description: ''
                        - value: V4_5ALL
                          name: ''
                          description: ''
                        - value: V5
                          name: ''
                          description: ''
                        - value: V5_5
                          name: ''
                          description: ''
                        - value: V6
                          name: ''
                          description: ''
                        - value: V6_MINI
                          name: ''
                          description: ''
                        - value: V6_WILD
                          name: ''
                          description: ''
                      examples:
                        - V6
                    negative_tags:
                      type: string
                      description: >-
                        Music styles or traits to exclude from the generated
                        audio. Optional. Use to avoid specific styles.
                      examples:
                        - Heavy Metal, Upbeat Drums
                    vocal_gender:
                      type: string
                      description: >-
                        Vocal gender preference. Optional. Use `m` for male and
                        `f` for female. This parameter is only effective when
                        `custom_mode` is true. In practice, it only increases
                        probability and cannot guarantee male/female voice
                        instructions are followed.
                      enum:
                        - m
                        - f
                      examples:
                        - m
                    style_weight:
                      type: number
                      description: >-
                        Only available when Custom Mode (`custom_mode: true`) is
                        enabled. Strength of adherence to the specified style.
                        Optional. Range 0–1, up to 2 decimal places.
                      minimum: 0
                      maximum: 1
                      multipleOf: 0.01
                      examples:
                        - 0.65
                    weirdness_constraint:
                      type: number
                      description: >-
                        Only available when Custom Mode (`custom_mode: true`) is
                        enabled. Controls experimental/creative deviation.
                        Optional. Range 0–1, up to 2 decimal places.
                      minimum: 0
                      maximum: 1
                      multipleOf: 0.01
                      examples:
                        - 0.65
                    audio_weight:
                      type: number
                      description: >-
                        Only available when Custom Mode (`custom_mode: true`) is
                        enabled. Relative weight of audio features. Optional.
                        Range 0–1, up to 2 decimal places.

                        - `audio_weight` is not supported when there are no
                        vocals
                      minimum: 0
                      maximum: 1
                      multipleOf: 0.01
                      examples:
                        - 0.65
                    persona_id:
                      type: string
                      description: >-
                        Persona ID or Voice ID to apply to the generated music.
                        Optional. Use this to apply a specific persona style to
                        your music generation.


                        To generate a Persona ID, use the [Generate
                        Persona](https://docs.kie.ai/suno-api/generate-persona)
                        endpoint to create a personalized music Persona based on
                        generated music.


                        To generate a Voice ID, use the [Generate
                        Voice](https://docs.kie.ai/suno-api/suno-voice-generate)
                        endpoint.
                      examples:
                        - persona_123
                    persona_model:
                      type: string
                      enum:
                        - style_persona
                        - voice_persona
                      x-apidog-enum:
                        - value: style_persona
                          name: ''
                          description: ''
                        - value: voice_persona
                          name: ''
                          description: ''
                      description: >-
                        Persona model. Only available for V5 (Discontinued),
                        V5.5 (Discontinued), V6, V6_MINI, and V6_WILD.
                    duration:
                      type: number
                      examples:
                        - 20
                      minimum: 10
                      maximum: 360
                      description: >-
                        Target length in seconds (10–360). Optional; if omitted,
                        the model chooses the length. Only valid when
                        `custom_mode` is true and the model is V5_5
                        (Discontinued), V6, V6_MINI, or V6_WILD.
                    image_urls:
                      type: array
                      description: >-
                        Image references. Only effective when
                        `custom_mode=false`.

                        - Optional

                        - Up to 5 images; each image must not exceed 10 MB

                        - Supported formats: `jpeg`, `png`, `webp`, and `bmp`
                      items:
                        type: string
                      maxItems: 5
                    variety:
                      type: number
                      default: 1
                      examples:
                        - 1
                      minimum: 0
                      maximum: 4
                      multipleOf: 1
                      description: >-
                        Only available when Custom Mode (`custom_mode: true`) is
                        enabled. Controls the diversity and stylistic variation
                        of generated results. Optional. Range 0–4, integer,
                        default 1.


                        - **`0`**: off (exact style) — fully off, strictly the
                        same style

                        - **`1`**: normal (balanced variety) — default, balances
                        stability and diversity

                        - **`2`**: high (distinct styles) — produces results
                        with clearly different styles

                        - **`3`**: extra (bold exploration) — higher variation,
                        encourages bold exploration of different styles

                        - **`4`**: max (unreasonably varied) — maximum
                        diversity; results may differ in style very
                        significantly
                    lyrics:
                      type: string
                      description: >-
                        Lyrics for the generated audio. Optional.

                        - V6: Maximum 5000 characters.

                        - When `custom_mode=true`, `lyrics` is used as lyrics
                        first; if `lyrics` is not provided, `prompt` is used as
                        lyrics.

                        - When `custom_mode=false`, `lyrics` is used as a lyrics
                        attachment and can be used together with the audio
                        description in `prompt`.
                    audio_urls:
                      type: array
                      items:
                        type: string
                      description: >-
                        Audio references. Only effective when
                        `custom_mode=false`.

                        - Optional

                        - Audio duration must be between 6 seconds and 30
                        minutes; each audio file must not exceed 500 MB
                    video_urls:
                      type: array
                      items:
                        type: string
                      description: >-
                        Video references. Only effective when
                        `custom_mode=false`.

                        - Optional

                        - Up to 1 video file; each file must not exceed 100 MB,
                        duration must not exceed 241 seconds, with no minimum
                        duration

                        - Supported formats: mp4, mov, and webm
                      maxItems: 1
                  required:
                    - custom_mode
                    - instrumental
                    - model
                  x-apidog-orders:
                    - prompt
                    - lyrics
                    - image_urls
                    - video_urls
                    - audio_urls
                    - style
                    - title
                    - custom_mode
                    - instrumental
                    - model
                    - negative_tags
                    - vocal_gender
                    - style_weight
                    - weirdness_constraint
                    - audio_weight
                    - variety
                    - persona_id
                    - persona_model
                    - duration
              x-apidog-orders:
                - model
                - callBackUrl
                - input
            example:
              model: ai-music-api/generate
              input:
                custom_mode: false
                instrumental: false
                model: V6
                prompt: An upbeat acoustic song about a summer road trip
      responses:
        '200':
          description: Request successful
          content:
            application/json:
              schema:
                allOf:
                  - type: object
                    properties:
                      code:
                        type: integer
                        enum:
                          - 200
                          - 401
                          - 402
                          - 404
                          - 409
                          - 422
                          - 429
                          - 451
                          - 455
                          - 500
                        description: >-
                          Status code of the request. The HTTP status is 200
                          even when the request fails, so always check this
                          field: 200 means success; any other value means the
                          request failed and msg describes the reason. Common
                          values: 401 invalid or missing API key; 402
                          insufficient credits; 429 rate limit exceeded; 422 or
                          500 the request could not be processed (read msg and
                          check your parameters before retrying).
                      msg:
                        type: string
                        description: Error message when code != 200
                        examples:
                          - success
                    x-apidog-orders:
                      - code
                      - msg
                  - type: object
                    properties:
                      data:
                        type: object
                        properties:
                          taskId:
                            type: string
                            description: >-
                              Task ID for tracking task status. Poll GET
                              /api/v1/jobs/recordInfo?taskId=... (see [Get Task
                              Details](/market/common/get-task-detail)). For
                              Suno audio tasks the tracks are in
                              data.response.data[] (audio_url, stream_audio_url,
                              image_url, id, duration).
                            examples:
                              - 5c79****be8e
                        x-apidog-orders:
                          - taskId
                    x-apidog-orders:
                      - data
          headers: {}
          x-apidog-name: ''
      security:
        - BearerAuth: []
          x-apidog:
            schemeGroups:
              - id: kn8M4YUlc5i0A0179ezwx
                schemeIds:
                  - BearerAuth
            required: true
            use:
              id: kn8M4YUlc5i0A0179ezwx
            scopes:
              kn8M4YUlc5i0A0179ezwx:
                BearerAuth: []
      callbacks:
        audioGenerated:
          '{request.body#/callBackUrl}':
            post:
              description: >-
                System will call this callback when audio generation is
                complete.


                ### Callback Example

                ```json

                {
                  "code": 200,
                  "msg": "All generated successfully.",
                  "data": {
                    "callbackType": "complete",
                    "task_id": "2fac****9f72",
                    "data": [
                      {
                        "id": "e231****-****-****-****-****8cadc7dc",
                        "audio_url": "https://example.cn/****.mp3",
                        "stream_audio_url": "https://example.cn/****",
                        "image_url": "https://example.cn/****.jpeg",
                        "prompt": "[Verse] Night city lights shining bright",
                        "model_name": "chirp-v3-5",
                        "title": "Iron Man",
                        "tags": "electrifying, rock",
                        "createTime": 1791200596437,
                        "duration": 198.44
                      },
                      {
                        "id": "bd15****1873",
                        "audio_url": "https://example.cn/****.mp3",
                        "stream_audio_url": "https://example.cn/****",
                        "image_url": "https://example.cn/****.jpeg",
                        "prompt": "[Verse] Night city lights shining bright",
                        "model_name": "chirp-v3-5",
                        "title": "Iron Man",
                        "tags": "electrifying, rock",
                        "createTime": 1791200596437,
                        "duration": 228.28
                      }
                    ]
                  }
                }

                ```
              requestBody:
                content:
                  application/json:
                    schema:
                      type: object
                      properties:
                        code:
                          type: integer
                          description: Status code
                          example: 200
                        msg:
                          type: string
                          description: Response message
                          example: All generated successfully
                        data:
                          type: object
                          properties:
                            callbackType:
                              type: string
                              description: >-
                                Callback type: text (text generation complete),
                                first (first track complete), complete (all
                                tracks complete)
                              enum:
                                - text
                                - first
                                - complete
                            task_id:
                              type: string
                              description: Task ID
                            data:
                              type: array
                              items:
                                type: object
                                properties:
                                  id:
                                    type: string
                                    description: Audio unique identifier (audioId)
                                  audio_url:
                                    type: string
                                    description: Audio file URL
                                  stream_audio_url:
                                    type: string
                                    description: Streaming audio URL
                                  image_url:
                                    type: string
                                    description: Cover image URL
                                  prompt:
                                    type: string
                                    description: Generation prompt/lyrics
                                  model_name:
                                    type: string
                                    description: Model name used
                                  title:
                                    type: string
                                    description: Music title
                                  tags:
                                    type: string
                                    description: Music tags
                                  createTime:
                                    type: integer
                                    description: >-
                                      Creation time (Unix timestamp in
                                      milliseconds)
                                  duration:
                                    type: number
                                    description: Audio duration (seconds)
              responses:
                '200':
                  description: Callback received successfully
                  content:
                    application/json:
                      schema:
                        allOf:
                          - type: object
                            properties:
                              code:
                                type: integer
                                enum:
                                  - 200
                                  - 400
                                  - 408
                                  - 413
                                  - 500
                                  - 501
                                  - 531
                                description: >-
                                  Status code. 200 means success; any other
                                  value means failure (for example, 501).
                              msg:
                                type: string
                                description: Error message when code != 200
                                example: success
                      example:
                        code: 200
                        msg: success
      x-apidog-folder: docs/en/Market/Music Models/Suno/Music Generation
      x-apidog-status: released
      x-run-in-apidog: https://app.apidog.com/web/project/1184766/apis/api-43206407-run
components:
  schemas: {}
  securitySchemes:
    BearerAuth:
      type: bearer
      scheme: bearer
      bearerFormat: API Key
      description: >-
        All API requests require a Bearer Token. Add the header `Authorization:
        Bearer YOUR_API_KEY` to authenticate requests.
    BearerAuth1:
      type: bearer
      scheme: bearer
      bearerFormat: API Key
      description: >-
        所有 API 请求都需要 Bearer Token。请在请求头中添加 `Authorization: Bearer YOUR_API_KEY`
        进行身份验证。
servers:
  - url: https://api.kie.ai
    description: 正式环境
security:
  - BearerAuth: []
    x-apidog:
      schemeGroups:
        - id: kn8M4YUlc5i0A0179ezwx
          schemeIds:
            - BearerAuth
      required: true
      use:
        id: kn8M4YUlc5i0A0179ezwx
      scopes:
        kn8M4YUlc5i0A0179ezwx:
          BearerAuth: []

```


# Suno Voice Create Custom Voice API

## OpenAPI Specification

```yaml
openapi: 3.0.1
info:
  title: ''
  description: ''
  version: 1.0.0
paths:
  /api/v1/jobs/createTask:
    post:
      summary: Suno Voice Create Custom Voice API
      deprecated: false
      description: >
        :::warning Document updated

        If you have already completed the integration process previously, you
        can still access the old version document at: Old version address
        (https://docs.kie.ai/old-model/suno-api/suno-voice-generate)

        :::

        Generate a custom Suno Voice after the user reads the validation phrase.


        Submit the original validation task ID, the user's verification audio
        URL, and optional voice metadata such as `voice_name`, `description`,
        and `style`. The system will validate the recording and create a
        reusable custom voice.


        For better results, upload or record a singing voice sample. Clean
        acapella vocals usually produce the most accurate voice profile.


        :::tip

        This endpoint is free.

        :::


        ## Query Task Status

        :::tip[]

        In the production environment, it is recommended to use the
        `callBackUrl` parameter to receive automatic notifications when the
        custom voice is generated, rather than polling the status endpoint.

        :::


        ## Use the Generated Voice


        Get the Voice ID: the Voice ID is the taskId of your validation-phrase
        task, which create-voice also returns in resultJson.resultUrls[0].
        create-voice returns success immediately; that only means the request
        was accepted. The voice is ready when the validation-phrase task changes
        to state "success" (usually within a minute). If it changes to "fail",
        the recording did not pass verification. Receive the result on the
        validation-phrase task's callBackUrl (data.status is "success" or
        "fail"), or poll Get Task Details with that taskId. To use the voice,
        pass the Voice ID as persona_id together with persona_model:
        "voice_persona" in Generate Music.


        Related resources


        <CardGroup cols={2}>
          <Card title="Check Voice Availability" icon="lucide-badge-check" href="/suno-api/suno-voice-check-voice"> Verify whether a generated voice is available for use </Card>
          <Card title="General API" icon="lucide-cog" href="/common-api/get-account-credits"> View account credits and usage </Card>
        </CardGroup>
      operationId: suno-voice-generate
      tags:
        - docs/en/Market/Music Models/Suno/voice
        - Suno API/voice
      parameters: []
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                model:
                  type: string
                  description: The model name to use for generation. Required field.
                  examples:
                    - ai-music-api/create-voice
                callBackUrl:
                  type: string
                  description: >-
                    The URL to receive generation task completion updates.
                    Optional but recommended for production use.


                    - System will POST task status and results to this URL when
                    generation completes

                    - Callback includes generated content URLs and task
                    information

                    - Your callback endpoint should accept POST requests with
                    JSON payload containing results

                    - Alternatively, use the Get Task Details endpoint to poll
                    task status

                    - To ensure callback security, see [Webhook Verification
                    Guide](/common-api/webhook-verification) for signature
                    verification implementation
                  format: uri
                input:
                  type: object
                  description: Input parameters for the generation task
                  properties:
                    task_id:
                      description: Task ID
                      type: string
                    verify_url:
                      description: >-
                        Audio URL for the user's recording of the
                        server-returned `validateInfo` validation phrase. For
                        best voice generation results, recording the phrase in a
                        singing voice rather than plain speech is recommended.
                        [Required]
                      type: string
                    voice_name:
                      description: Voice name
                      type: string
                    description:
                      description: Voice description
                      type: string
                    style:
                      description: Voice style
                      type: string
                    singer_skill_level:
                      type: string
                      description: >-
                        Singer skill level. Supported: beginner, intermediate,
                        advanced, professional
                      default: beginner
                      enum:
                        - beginner
                        - intermediate
                        - advanced
                        - professional
                      x-apidog-enum:
                        - value: beginner
                          name: ''
                          description: ''
                        - value: intermediate
                          name: ''
                          description: ''
                        - value: advanced
                          name: ''
                          description: ''
                        - value: professional
                          name: ''
                          description: ''
                  required:
                    - task_id
                    - verify_url
                  x-apidog-orders:
                    - task_id
                    - verify_url
                    - voice_name
                    - description
                    - style
                    - singer_skill_level
              required:
                - model
                - input
              x-apidog-orders:
                - model
                - callBackUrl
                - input
            example:
              model: ai-music-api/create-voice
              callBackUrl: https://webbed-kit.name/
              input:
                task_id: '12'
                verify_url: https://vengeful-provider.org/
                voice_name: Doreen Murphy
                description: >-
                  Calco charisma cursus. Vulgus arbitro uberrime congregatio
                  assentator demoror conor velociter calculus crux. Deprimo
                  varius appello deputo voveo carus quas cohors corporis pel.
                  Sponte adinventitias bellicus veniam. Colligo angelus abscido
                  vos caveo. Aurum creber toties talio animi. Truculenter
                  exercitationem cena inventore. Blandior vereor verumtamen
                  demoror iure caries tonsor cruentus.
                style: laboris
                singer_skill_level: professional
      responses:
        '200':
          description: Request successful.
          content:
            application/json:
              schema:
                type: object
                properties:
                  code:
                    type: integer
                    description: >-
                      Status code of the request. The HTTP status is 200 even
                      when the request fails, so always check this field: 200
                      means success; any other value means the request failed
                      and msg describes the reason. Common values: 401 invalid
                      or missing API key; 402 insufficient credits; 429 rate
                      limit exceeded; 422 or 500 the request could not be
                      processed (read msg and check your parameters before
                      retrying).
                  msg:
                    type: string
                  data:
                    type: object
                    properties:
                      taskId:
                        description: Task ID
                        type: string
                    required:
                      - taskId
                    x-apidog-orders:
                      - taskId
                required:
                  - code
                  - msg
                  - data
                x-apidog-orders:
                  - code
                  - msg
                  - data
              example:
                code: 200
                msg: success
                data:
                  taskId: xxx_task_id_xxx
          headers: {}
          x-apidog-name: ''
      security:
        - BearerAuth: []
          x-apidog:
            schemeGroups:
              - id: kn8M4YUlc5i0A0179ezwx
                schemeIds:
                  - BearerAuth
            required: true
            use:
              id: kn8M4YUlc5i0A0179ezwx
            scopes:
              kn8M4YUlc5i0A0179ezwx:
                BearerAuth: []
      x-apidog-folder: docs/en/Market/Music Models/Suno/voice
      x-apidog-status: released
      x-run-in-apidog: https://app.apidog.com/web/project/1184766/apis/api-43206409-run
components:
  schemas: {}
  securitySchemes:
    BearerAuth:
      type: bearer
      scheme: bearer
      bearerFormat: API Key
      description: >-
        All API requests require a Bearer Token. Add the header `Authorization:
        Bearer YOUR_API_KEY` to authenticate requests.
    BearerAuth1:
      type: bearer
      scheme: bearer
      bearerFormat: API Key
      description: >-
        所有 API 请求都需要 Bearer Token。请在请求头中添加 `Authorization: Bearer YOUR_API_KEY`
        进行身份验证。
servers:
  - url: https://api.kie.ai
    description: 正式环境
security:
  - BearerAuth: []
    x-apidog:
      schemeGroups:
        - id: kn8M4YUlc5i0A0179ezwx
          schemeIds:
            - BearerAuth
      required: true
      use:
        id: kn8M4YUlc5i0A0179ezwx
      scopes:
        kn8M4YUlc5i0A0179ezwx:
          BearerAuth: []

```

# Generate sounds

## OpenAPI Specification

```yaml
openapi: 3.0.1
info:
  title: ''
  description: ''
  version: 1.0.0
paths:
  /api/v1/jobs/createTask:
    post:
      summary: Generate sounds
      deprecated: false
      description: >
        :::warning Document updated

        If you have already completed the integration process previously, you
        can still access the old version document at: Old version address
        (https://docs.kie.ai/old-model/suno-api/generate-sounds)

        :::

        Used for creating a sound generation task (Sounds Task). It supports
        settings for looping, tempo (BPM), pitch (Key), as well as lyrics
        subtitle capture, etc. 


        ## 🚀 User Guide 

        - By using this interface, you can generate corresponding audio content
        based on the input `prompt`.

        - It supports setting up loop playback effect, which is suitable for
        background music, ambient sounds, and other scenarios.

        - It allows specifying BPM (beats per minute) and pitch (Key) to
        facilitate control over the style of the generated result.

        - Optional feature to enable lyric subtitle capture for easier display
        or processing of lyric content later.

        - Supports asynchronous reception of task completion notifications
        through callBackUrl address. 


        ## 📌 Usage Scenarios

        - 🎧 Background music creation

        - 🎮 Game sound effects or looped ambient sounds generation

        - 🌐 Integration of audio content platforms and creative tools


        When polling GET /api/v1/jobs/recordInfo, the clips are in
        data.response.data[] (audio_url, stream_audio_url, image_url, id,
        duration).
      operationId: generate-sounds
      tags:
        - docs/en/Market/Music Models/Suno/Sounds Generation
      parameters: []
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                model:
                  type: string
                  description: The model name to use for generation. Required field.
                  examples:
                    - ai-music-api/sounds
                callBackUrl:
                  type: string
                  title: Callback address
                  description: >-
                    The URL to receive generation task completion updates.
                    Optional but recommended for production use.


                    - System will POST task status and results to this URL when
                    generation completes

                    - Callback includes generated content URLs and task
                    information

                    - Your callback endpoint should accept POST requests with
                    JSON payload containing results

                    - Alternatively, use the Get Task Details endpoint to poll
                    task status

                    - To ensure callback security, see [Webhook Verification
                    Guide](/common-api/webhook-verification) for signature
                    verification implementation
                input:
                  type: object
                  description: Input parameters for the generation task
                  properties:
                    prompt:
                      description: Description of the sound (max 500 characters).
                      type: string
                      title: prompt
                    model:
                      type: string
                      title: ''
                      description: >-
                        Required. Use V6, V6_MINI or V6_WILD.

                        - **`V6_WILD`**: Pushes creative boundaries for bolder,
                        more distinctive musical expression.

                        - **`V6_MINI`**: Lightweight and fast, balancing quality
                        and speed for effortless creation.

                        - **`V6`**: Greater musical expression with more natural
                        vocals and richer details.
                      enum:
                        - V5
                        - V5_5
                        - V6
                        - V6_WILD
                        - V6_MINI
                      x-apidog-enum:
                        - value: V5
                          name: ''
                          description: ''
                        - value: V5_5
                          name: ''
                          description: ''
                        - value: V6
                          name: ''
                          description: ''
                        - value: V6_WILD
                          name: ''
                          description: ''
                        - value: V6_MINI
                          name: ''
                          description: ''
                    sound_loop:
                      type: boolean
                      title: ''
                      description: Whether the generated sound loops seamlessly.
                      default: false
                    sound_tempo:
                      type: integer
                      title: BPM:Beats per minute
                      description: Tempo in BPM (1–300). Omit for automatic tempo.
                      default: null
                      minimum: 1
                      maximum: 300
                    sound_key:
                      type: string
                      title: ''
                      default: Any
                      enum:
                        - Any
                        - Cm
                        - C#m
                        - Dm
                        - D#m
                        - Em
                        - Fm
                        - F#m
                        - Gm
                        - G#m
                        - Am
                        - A#m
                        - Bm
                        - C
                        - C#
                        - D
                        - D#
                        - E
                        - F
                        - F#
                        - G
                        - G#
                        - A
                        - A#
                        - B
                      x-apidog-enum:
                        - value: Any
                          name: ''
                          description: ''
                        - value: Cm
                          name: ''
                          description: ''
                        - value: C#m
                          name: ''
                          description: ''
                        - value: Dm
                          name: ''
                          description: ''
                        - value: D#m
                          name: ''
                          description: ''
                        - value: Em
                          name: ''
                          description: ''
                        - value: Fm
                          name: ''
                          description: ''
                        - value: F#m
                          name: ''
                          description: ''
                        - value: Gm
                          name: ''
                          description: ''
                        - value: G#m
                          name: ''
                          description: ''
                        - value: Am
                          name: ''
                          description: ''
                        - value: A#m
                          name: ''
                          description: ''
                        - value: Bm
                          name: ''
                          description: ''
                        - value: C
                          name: ''
                          description: ''
                        - value: C#
                          name: ''
                          description: ''
                        - value: D
                          name: ''
                          description: ''
                        - value: D#
                          name: ''
                          description: ''
                        - value: E
                          name: ''
                          description: ''
                        - value: F
                          name: ''
                          description: ''
                        - value: F#
                          name: ''
                          description: ''
                        - value: G
                          name: ''
                          description: ''
                        - value: G#
                          name: ''
                          description: ''
                        - value: A
                          name: ''
                          description: ''
                        - value: A#
                          name: ''
                          description: ''
                        - value: B
                          name: ''
                          description: ''
                      examples:
                        - Any
                      description: Musical key. Default "Any" (automatic).
                    grab_lyrics:
                      type: boolean
                      title: Grab the lyrics subtitles
                      description: >-
                        Whether to capture the lyrics subtitles 


                        Will the interface be called after completion to obtain
                        the lyrics subtitles?
                      default: false
                  required:
                    - prompt
                    - model
                  x-apidog-orders:
                    - prompt
                    - model
                    - sound_loop
                    - sound_tempo
                    - sound_key
                    - grab_lyrics
              required:
                - model
                - input
              x-apidog-orders:
                - model
                - callBackUrl
                - input
            example:
              model: ai-music-api/sounds
              callBackUrl: https://api.example.com/callback
              input:
                prompt: sint
                model: V6
                sound_loop: true
                sound_tempo: 166
                sound_key: D#m
                grab_lyrics: true
      responses:
        '200':
          description: Request successful
          content:
            application/json:
              schema:
                type: object
                required:
                  - data
                properties:
                  code:
                    type: integer
                    description: >-
                      Status code of the request. The HTTP status is 200 even
                      when the request fails, so always check this field: 200
                      means success; any other value means the request failed
                      and msg describes the reason. Common values: 401 invalid
                      or missing API key; 402 insufficient credits; 429 rate
                      limit exceeded; 422 or 500 the request could not be
                      processed (read msg and check your parameters before
                      retrying).
                    enum:
                      - 200
                      - 401
                      - 402
                      - 404
                      - 422
                      - 429
                      - 433
                      - 455
                      - 500
                      - 501
                      - 505
                  msg:
                    type: string
                    description: Response message, error description upon failure
                    examples:
                      - success
                  data:
                    type: object
                    required:
                      - taskId
                    properties:
                      taskId:
                        type: string
                        description: >-
                          The task ID can be used with the "Get Task Details"
                          endpoint to query the task status.
                        examples:
                          - dc1928bfcbc77cb6c85f3359a9c718b3
                    x-apidog-orders:
                      - taskId
                x-apidog-orders:
                  - code
                  - msg
                  - data
              example:
                code: 200
                msg: success
                data:
                  taskId: task38695****935u9043u
          headers: {}
          x-apidog-name: ''
      security:
        - BearerAuth: []
          x-apidog:
            schemeGroups:
              - id: kn8M4YUlc5i0A0179ezwx
                schemeIds:
                  - BearerAuth
            required: true
            use:
              id: kn8M4YUlc5i0A0179ezwx
            scopes:
              kn8M4YUlc5i0A0179ezwx:
                BearerAuth: []
      x-apidog-folder: docs/en/Market/Music Models/Suno/Sounds Generation
      x-apidog-status: released
      x-run-in-apidog: https://app.apidog.com/web/project/1184766/apis/api-43206422-run
components:
  schemas: {}
  securitySchemes:
    BearerAuth:
      type: bearer
      scheme: bearer
      bearerFormat: API Key
      description: >-
        All API requests require a Bearer Token. Add the header `Authorization:
        Bearer YOUR_API_KEY` to authenticate requests.
    BearerAuth1:
      type: bearer
      scheme: bearer
      bearerFormat: API Key
      description: >-
        所有 API 请求都需要 Bearer Token。请在请求头中添加 `Authorization: Bearer YOUR_API_KEY`
        进行身份验证。
servers:
  - url: https://api.kie.ai
    description: 正式环境
security:
  - BearerAuth: []
    x-apidog:
      schemeGroups:
        - id: kn8M4YUlc5i0A0179ezwx
          schemeIds:
            - BearerAuth
      required: true
      use:
        id: kn8M4YUlc5i0A0179ezwx
      scopes:
        kn8M4YUlc5i0A0179ezwx:
          BearerAuth: []

```


# Generate Persona

## OpenAPI Specification

```yaml
openapi: 3.0.1
info:
  title: ''
  description: ''
  version: 1.0.0
paths:
  /api/v1/jobs/createTask:
    post:
      summary: Generate Persona
      deprecated: false
      description: >-
        :::warning Document updated

        If you have already completed the integration process previously, you
        can still access the old version document at: Old version address
        (https://docs.kie.ai/old-model/suno-api/generate-persona)

        :::

        > Create a personalized music Persona based on generated music, giving
        the music a unique identity and characteristics.


        ## Usage Guide


        Use this endpoint to create Personas (music characters) for generated
        music:

        * Requires the `task_id` from music generation related endpoints
        (generate, extend) and audio ID

        * Customize the Persona name and description to give music unique
        personality

        * Generated Personas can be used for subsequent music creation and style
        transfer


        ## Parameter Details


        *   **`task_id`** (Required): Can be obtained from the following
        endpoints:
            *   Generate Music
            *   Extend Music
        *   **`audio_id`** (Required): Specifies the audio ID to create Persona
        for

        *   **`name`** (Required): Assigns an easily recognizable name to the
        Persona

        *   **`description`** (Required): Describes the Persona's musical
        characteristics, style, and personality


        ## Developer Notes


        :::caution Important Requirements

        *   **Ensure the music generation task is fully completed** before
        calling this endpoint. If the music is still generating, this endpoint
        will return a failure.

        *   Each audio ID can only generate a Persona **once**.

        :::


        *   It is recommended to provide detailed descriptions for Personas to
        better capture musical characteristics.

        *   The returned `persona_id` can be used in subsequent music generation
        requests to create music with similar style characteristics.

        *   You can apply the `persona_id` to the following endpoints:
            *   Generate Music
            *   Extend Music

        ## Parameter Example


        ```json

        {
          "task_id": "5c79****be8e",
          "audio_id": "e231****-****-****-****-****8cadc7dc",
          "name": "Electronic Pop Singer",
          "description": "A modern electronic music style pop singer, skilled in dynamic rhythms and synthesizer tones"
        }

        ```


        :::note

        Ensure that the music generation task corresponding to the `task_id` is
        complete and the `audio_id` is within the valid range.

        :::


        :::tip

        Providing detailed and specific descriptions for Personas helps the
        system more accurately capture musical style characteristics.

        :::
      operationId: generate-persona
      tags:
        - docs/en/Market/Music Models/Suno/Music Generation
      parameters: []
      requestBody:
        content:
          application/json:
            schema:
              type: object
              required:
                - model
                - input
              properties:
                model:
                  type: string
                  description: The model name to use for generation. Required field.
                  examples:
                    - ai-music-api/generate-persona
                callBackUrl:
                  type: string
                  description: >-
                    The URL to receive generation task completion updates.
                    Optional but recommended for production use.


                    - System will POST task status and results to this URL when
                    generation completes

                    - Callback includes generated content URLs and task
                    information

                    - Your callback endpoint should accept POST requests with
                    JSON payload containing results

                    - Alternatively, poll GET /api/v1/jobs/recordInfo?taskId=...
                    (see [Get Task Details](/market/common/get-task-detail)).

                    - To ensure callback security, see [Webhook Verification
                    Guide](/common-api/webhook-verification) for signature
                    verification implementation
                  format: uri
                input:
                  type: object
                  description: Input parameters for the generation task
                  properties:
                    task_id:
                      type: string
                      description: >-
                        Unique identifier of the original music generation task.
                        Generated by model ai-music-api/generate or
                        ai-music-api/extend.
                      examples:
                        - 5c79****be8e
                    audio_id:
                      type: string
                      description: >-
                        Unique identifier of the audio track to create Persona
                        for. This ID is returned in the callBackUrl data after
                        music generation completes.
                      examples:
                        - e231****-****-****-****-****8cadc7dc
                    name:
                      type: string
                      description: >-
                        Name for the Persona. A descriptive name that captures
                        the essence of the musical style or character.
                      examples:
                        - Electronic Pop Singer
                    description:
                      type: string
                      description: >-
                        Detailed description of the Persona's musical
                        characteristics, style, and personality. Be specific
                        about genre, mood, instrumentation, and vocal qualities.
                      examples:
                        - >-
                          A modern electronic music style pop singer, skilled in
                          dynamic rhythms and synthesizer tones
                    vocal_start:
                      type: number
                      description: >-
                        Start time (in seconds) for Persona analysis segment
                        extraction. Used to specify the time point in the audio
                        from which to extract the segment for Persona analysis.
                        Must be less than vocal_end, and vocal_end - vocal_start
                        must be between 10–30 seconds. Defaults to 0.0.
                      default: 0
                      examples:
                        - 12.5
                      minimum: 0
                    vocal_end:
                      type: number
                      description: >-
                        End time (in seconds) for Persona analysis segment
                        extraction. Together with vocal_start, used to specify
                        the time range for analysis. vocal_end - vocal_start
                        must be between 10–30 seconds. Defaults to 30.0.
                      multipleOf: 0.01
                      default: 30
                      examples:
                        - 25.8
                      minimum: 0
                    style:
                      type: string
                      description: >-
                        Optional. Used to supplement the description of the
                        music style tag corresponding to the Persona, such as
                        "Electronic Pop", "Jazz Trio", etc.
                  required:
                    - task_id
                    - audio_id
                    - name
                    - description
                  x-apidog-orders:
                    - task_id
                    - audio_id
                    - name
                    - description
                    - vocal_start
                    - vocal_end
                    - style
                  x-apidog-ignore-properties: []
              x-apidog-orders:
                - model
                - callBackUrl
                - input
              x-apidog-ignore-properties: []
            example:
              model: ai-music-api/generate-persona
              callBackUrl: https://api.example.com/callback
              input:
                task_id: 5c79****be8e
                audio_id: e231****-****-****-****-****8cadc7dc
                name: Electronic Pop Singer
                description: >-
                  A modern electronic music style pop singer, skilled in dynamic
                  rhythms and synthesizer tones
                vocal_start: 0
                vocal_end: 30
                style: Electronic Pop
      responses:
        '200':
          description: Request successful
          content:
            application/json:
              schema:
                allOf:
                  - type: object
                    properties:
                      code:
                        type: integer
                        enum:
                          - 200
                          - 401
                          - 402
                          - 404
                          - 409
                          - 422
                          - 429
                          - 451
                          - 455
                          - 500
                        description: >-
                          Status code of the request. The HTTP status is 200
                          even when the request fails, so always check this
                          field: 200 means success; any other value means the
                          request failed and msg describes the reason. Common
                          values: 401 invalid or missing API key; 402
                          insufficient credits; 429 rate limit exceeded; 422 or
                          500 the request could not be processed (read msg and
                          check your parameters before retrying).
                      msg:
                        type: string
                        description: Error message when code != 200
                        examples:
                          - success
                    x-apidog-orders:
                      - code
                      - msg
                    x-apidog-ignore-properties: []
                  - type: object
                    properties:
                      data:
                        type: object
                        properties:
                          persona_id:
                            type: string
                            description: >-
                              Unique identifier for the generated Persona. This
                              persona_id can be used in subsequent music
                              generation requests (Generate Music, Extend Music,
                              Upload And Cover Audio, Upload And Extend Audio)
                              to create music with similar style
                              characteristics.
                            examples:
                              - a1b2****c3d4
                          name:
                            type: string
                            description: Name of the Persona as provided in the request.
                            examples:
                              - Electronic Pop Singer
                          description:
                            type: string
                            description: >-
                              Description of the Persona's musical
                              characteristics, style, and personality as
                              provided in the request.
                            examples:
                              - >-
                                A modern electronic music style pop singer,
                                skilled in dynamic rhythms and synthesizer tones
                        x-apidog-orders:
                          - persona_id
                          - name
                          - description
                        x-apidog-ignore-properties: []
                    x-apidog-orders:
                      - data
                    x-apidog-ignore-properties: []
          headers: {}
          x-apidog-name: ''
        '500':
          description: Server error.
          content:
            application/json:
              schema:
                type: object
                properties:
                  code:
                    type: integer
                    description: >-
                      Response status code


                      - **200**: Success - Request has been processed
                      successfully

                      - **401**: Unauthorized - Authentication credentials are
                      missing or invalid

                      - **402**: Insufficient Credits - Account does not have
                      enough credits to perform the operation

                      - **404**: Not Found - The requested resource or endpoint
                      does not exist

                      - **408**: Timeout - No result has been returned for over
                      10 minutes.

                      - **422**: Validation Error - The request parameters
                      failed validation checks

                      - **429**: Rate Limited - Request limit has been exceeded
                      for this resource

                      - **455**: Service Unavailable - System is currently
                      undergoing maintenance

                      - **500**: Server Error - An unexpected error occurred
                      while processing the request

                      - **501**: Generation Failed - Content generation task
                      failed

                      - **505**: Feature Disabled - The requested feature is
                      currently disabled
                  msg:
                    type: string
                    description: Response message, error description when failed
                  data:
                    type: object
                    properties: {}
                    x-apidog-orders: []
                    x-apidog-ignore-properties: []
                x-apidog-orders:
                  - code
                  - msg
                  - data
                required:
                  - code
                  - msg
                  - data
                x-apidog-ignore-properties: []
              example:
                code: 500
                msg: >-
                  Server Error - An unexpected error occurred while processing
                  the request
                data: null
          headers: {}
          x-apidog-name: Error
      security:
        - BearerAuth: []
          x-apidog:
            schemeGroups:
              - id: kn8M4YUlc5i0A0179ezwx
                schemeIds:
                  - BearerAuth
            required: true
            use:
              id: kn8M4YUlc5i0A0179ezwx
            scopes:
              kn8M4YUlc5i0A0179ezwx:
                BearerAuth: []
      x-apidog-folder: docs/en/Market/Music Models/Suno/Music Generation
      x-apidog-status: released
      x-run-in-apidog: https://app.apidog.com/web/project/1184766/apis/api-43206431-run
components:
  schemas: {}
  securitySchemes:
    BearerAuth:
      type: bearer
      scheme: bearer
      bearerFormat: API Key
      description: >-
        All API requests require a Bearer Token. Add the header `Authorization:
        Bearer YOUR_API_KEY` to authenticate requests.
    BearerAuth1:
      type: bearer
      scheme: bearer
      bearerFormat: API Key
      description: >-
        所有 API 请求都需要 Bearer Token。请在请求头中添加 `Authorization: Bearer YOUR_API_KEY`
        进行身份验证。
servers:
  - url: https://api.kie.ai
    description: 正式环境
security:
  - BearerAuth: []
    x-apidog:
      schemeGroups:
        - id: kn8M4YUlc5i0A0179ezwx
          schemeIds:
            - BearerAuth
      required: true
      use:
        id: kn8M4YUlc5i0A0179ezwx
      scopes:
        kn8M4YUlc5i0A0179ezwx:
          BearerAuth: []

```


# Replace Music Section

## OpenAPI Specification

```yaml
openapi: 3.0.1
info:
  title: ''
  description: ''
  version: 1.0.0
paths:
  /api/v1/jobs/createTask:
    post:
      summary: Replace Music Section
      deprecated: false
      description: >
        :::warning Document updated

        If you have already completed the integration process previously, you
        can still access the old version document at: Old version address
        (https://docs.kie.ai/old-model/suno-api/replace-section)

        :::

        > Replace a specific time segment within existing music.


        This interface can replace specific time segments in already generated
        music. It requires providing the original music's task ID and the time
        range to be replaced. The replaced audio will naturally blend with the
        original music. This endpoint is always custom mode.


        ### Parameter Details


        * Always required: `prompt`, `tags`, `title`, `infill_start_s`,
        `infill_end_s`, `full_lyrics`

        * `lyrics` is optional lyrics for the replaced segment. V6 maximum 5000
        characters

        * Optional: `negative_tags`, `vocal_gender`, `style_weight`,
        `weirdness_constraint`, `audio_weight`, `variety`, `persona_id`,
        `persona_model`


        ### Time Range Instructions


        *   `infill_start_s` must be less than `infill_end_s`.

        *   Time values are precise to 2 decimal places, e.g., `10.50` seconds.

        *   The replacement time must be at least **10 seconds**.

        *   Replacement duration should not exceed **50%** of the original
        music's total duration.


        ### Developer Notes


        *   Replacement segments will be regenerated based on the provided
        `prompt` and `tags`.

        *   Generated replacement segments will automatically blend with the
        original music's preceding and following parts.

        *   Generated files will be retained for **14 days**.
      operationId: replace-section
      tags:
        - docs/en/Market/Music Models/Suno/Music Generation
      parameters: []
      requestBody:
        content:
          application/json:
            schema:
              type: object
              required:
                - model
                - input
              properties:
                model:
                  type: string
                  description: >-
                    The model name used to generate the task. This field is
                    required. This endpoint must use the
                    `ai-music-api/replace-section` model.
                  enum:
                    - ai-music-api/replace-section
                  examples:
                    - ai-music-api/replace-section
                  x-apidog-enum:
                    - value: ai-music-api/replace-section
                      name: ''
                      description: ''
                callBackUrl:
                  type: string
                  format: uri
                  description: >-
                    The URL to receive updates when the generation task is
                    completed. Optional, but recommended for production
                    environments.


                    - When generation is complete, the system will POST the task
                    status and results to this URL

                    - The callback includes the generated URL and task
                    information

                    - Your callback endpoint should accept POST requests with a
                    JSON payload containing the results

                    - Alternatively, poll GET /api/v1/jobs/recordInfo?taskId=...
                    (see [Get Task Details](/market/common/get-task-detail)).
                    For Suno audio tasks the tracks are in data.response.data[]
                    (audio_url, stream_audio_url, image_url, id, duration).

                    - To ensure callback security, refer to the [Webhook
                    Verification Guide](/common-api/webhook-verification) for
                    signature verification implementation
                  examples:
                    - https://api.example.com/callback
                input:
                  type: object
                  description: Input parameters for the music section replacement task.
                  required:
                    - prompt
                    - tags
                    - title
                    - infill_start_s
                    - infill_end_s
                    - full_lyrics
                  oneOf:
                    - title: Replace section using existing audio
                      required:
                        - task_id
                        - audio_id
                      not:
                        anyOf:
                          - required:
                              - upload_url
                          - required:
                              - model
                      properties:
                        task_id:
                          type: string
                          description: >-
                            The original task ID (parent task), used to identify
                            the source music for section replacement.
                          examples:
                            - 2fac****9f72
                        audio_id:
                          type: string
                          description: >-
                            The unique identifier of the audio track to be
                            replaced. This ID is returned in the callback data
                            after music generation is completed.
                          examples:
                            - e231****-****-****-****-****8cadc7dc
                      x-apidog-orders:
                        - task_id
                        - audio_id
                      type: object
                    - title: Replace section using uploaded custom audio
                      required:
                        - upload_url
                        - model
                      not:
                        anyOf:
                          - required:
                              - task_id
                          - required:
                              - audio_id
                      properties:
                        upload_url:
                          type: string
                          format: uri
                          description: The URL of the custom audio uploaded by the user.
                          examples:
                            - https://example.com/audio.mp3
                        model:
                          type: string
                          default: V6
                          description: |-
                            The AI model version used for generation. 
                            - Available options:
                              - **`V6_WILD`**: Breaks through stylistic boundaries, delivering bolder and more unique musical expression.
                              - **`V6_MINI`**: Lightweight and fast, balancing audio quality and efficiency for easier inspirational creation.
                              - **`V6`**: Stronger musical expressiveness, more natural vocals, and richer details
                              - **`V5_5`** (deprecated): A dedicated model tailored to your unique taste.   
                              - **`V5`** (deprecated): Superior musical expressiveness with faster generation.  
                              - **`V4_5PLUS`** (deprecated): V4.5+ offers richer timbres, new creative methods, and supports durations of up to 8 minutes.  
                              - **`V4_5`** (deprecated): V4.5 provides smarter prompts, faster generation, and supports durations of up to 8 minutes.  
                              - **`V4_5ALL`** (deprecated): V4.5ALL provides smarter prompts, faster generation, and supports durations of up to 8 minutes.  
                              - **`V4`** (deprecated): V4 offers improved vocal quality and supports durations of up to 4 minutes.
                          enum:
                            - V4
                            - V4_5
                            - V4_5PLUS
                            - V4_5ALL
                            - V5
                            - V5_5
                            - V6
                            - V6_MINI
                            - V6_WILD
                          examples:
                            - V6
                          x-apidog-enum:
                            - value: V4
                              name: ''
                              description: ''
                            - value: V4_5
                              name: ''
                              description: ''
                            - value: V4_5PLUS
                              name: ''
                              description: ''
                            - value: V4_5ALL
                              name: ''
                              description: ''
                            - value: V5
                              name: ''
                              description: ''
                            - value: V5_5
                              name: ''
                              description: ''
                            - value: V6
                              name: ''
                              description: ''
                            - value: V6_MINI
                              name: ''
                              description: ''
                            - value: V6_WILD
                              name: ''
                              description: ''
                      x-apidog-orders:
                        - upload_url
                        - model
                      type: object
                  properties:
                    prompt:
                      type: string
                      description: The lyrics after replacement. Required.
                      examples:
                        - A calm and relaxing piano track.
                    tags:
                      type: string
                      description: >-
                        Style tags for the music, such as jazz or electronic.
                        Required.
                      examples:
                        - Jazz
                    title:
                      type: string
                      description: The title of the music. Required.
                      examples:
                        - Relaxing Piano
                    negative_tags:
                      type: string
                      description: >-
                        Music styles to exclude, used to prevent specific
                        stylistic elements from appearing in the replacement
                        section.
                      examples:
                        - Rock
                    infill_start_s:
                      type: number
                      description: >-
                        The point in time, in seconds, at which replacement
                        starts, rounded to 2 decimal places. Required. Must be
                        less than infill_end_s, and the difference between
                        infill_end_s and infill_start_s must be at least 10
                        seconds.
                      minimum: 0
                      examples:
                        - 10.5
                    infill_end_s:
                      type: number
                      description: >-
                        The point in time, in seconds, at which replacement
                        ends, rounded to 2 decimal places. Required. Must be
                        greater than infill_start_s, and the difference between
                        infill_end_s and infill_start_s must be at least 10
                        seconds.
                      minimum: 0
                      examples:
                        - 20.75
                    full_lyrics:
                      type: string
                      description: >-
                        The complete modified lyrics, combining the modified and
                        unmodified lyrics. Required. This parameter contains the
                        complete lyrics text that will be used for the entire
                        song after section replacement.
                      examples:
                        - >-
                          [Verse 1]\nOriginal lyrics content\n[Chorus]\nModified
                          lyrics for this section\n[Verse 2]\nMore original
                          lyrics
                    lyrics:
                      type: string
                      maxLength: 5000
                      description: |-
                        Lyrics for the replaced segment. Optional.
                        - V6, V6_MINI, and V6_WILD: Maximum 5000 characters.
                      examples:
                        - '[Chorus]\nModified lyrics for this section'
                    vocal_gender:
                      type: string
                      description: >-
                        Vocal gender preference. Optional. 'm' for male, 'f' for
                        female. Based on practice, this parameter can only
                        increase the probability but cannot guarantee adherence
                        to male/female voice instructions.
                      enum:
                        - m
                        - f
                      x-apidog-enum:
                        - value: m
                          name: ''
                          description: ''
                        - value: f
                          name: ''
                          description: ''
                      examples:
                        - m
                    style_weight:
                      type: number
                      description: >-
                        Adherence strength to specified style. Optional. Range
                        0–1, up to 2 decimal places.
                      minimum: 0
                      maximum: 1
                      multipleOf: 0.01
                      examples:
                        - 0.65
                    weirdness_constraint:
                      type: number
                      description: >-
                        Controls experimental/creative deviation level.
                        Optional. Range 0–1, up to 2 decimal places.
                      minimum: 0
                      maximum: 1
                      multipleOf: 0.01
                      examples:
                        - 0.65
                    audio_weight:
                      type: number
                      description: >-
                        Relative weight of audio elements. Optional. Range 0–1,
                        up to 2 decimal places.
                      minimum: 0
                      maximum: 1
                      multipleOf: 0.01
                      examples:
                        - 0.65
                    variety:
                      type: integer
                      default: 1
                      examples:
                        - 1
                      minimum: 0
                      maximum: 4
                      description: >-
                        Controls the diversity and stylistic variation of
                        generated results. Optional. Integer range 0–4, default
                        1.


                        - **`0`**: off (exact style) — fully off, strictly the
                        same style

                        - **`1`**: normal (balanced variety) — default, balances
                        stability and diversity

                        - **`2`**: high (distinct styles) — produces results
                        with clearly different styles

                        - **`3`**: extra (bold exploration) — higher variation,
                        encourages bold exploration of different styles

                        - **`4`**: max (unreasonably varied) — maximum
                        diversity; results may differ in style very
                        significantly
                    persona_id:
                      type: string
                      description: >-
                        Persona ID or Voice ID to apply to the generated music.
                        Optional. Use this to apply a specific persona style to
                        your music generation.


                        To generate a Persona ID, use the [Generate
                        Persona](https://docs.kie.ai/suno-api/generate-persona)
                        endpoint to create a personalized music Persona based on
                        generated music.


                        To generate a Voice ID, use the [Generate
                        Voice](https://docs.kie.ai/suno-api/suno-voice-generate)
                        endpoint.
                      examples:
                        - persona_123
                    persona_model:
                      type: string
                      enum:
                        - style_persona
                        - voice_persona
                      x-apidog-enum:
                        - value: style_persona
                          name: ''
                          description: ''
                        - value: voice_persona
                          name: ''
                          description: ''
                      default: style_persona
                      description: >-
                        Persona model. Optional. Use `style_persona` for
                        Generate Persona IDs, or `voice_persona` for Suno Voice
                        IDs. Default: `style_persona`. Available for V5
                        (Discontinued), V5_5 (Discontinued), V6, V6_MINI, and
                        V6_WILD.
                  x-apidog-orders:
                    - prompt
                    - lyrics
                    - tags
                    - title
                    - negative_tags
                    - infill_start_s
                    - infill_end_s
                    - full_lyrics
                    - vocal_gender
                    - style_weight
                    - weirdness_constraint
                    - audio_weight
                    - variety
                    - persona_id
                    - persona_model
              x-apidog-orders:
                - model
                - callBackUrl
                - input
            example:
              model: ai-music-api/replace-section
              callBackUrl: https://example.com/callback
              input:
                task_id: 2fac****9f72
                audio_id: e231****-****-****-****-****8cadc7dc
                prompt: A calm and relaxing piano track.
                tags: Jazz
                title: Relaxing Piano
                negative_tags: Rock
                infill_start_s: 10.5
                infill_end_s: 20.75
                full_lyrics: |-
                  [Verse 1]
                  Original lyrics here
                  [Chorus]
                  Modified lyrics for this section
                  [Verse 2]
                  More original lyrics
                lyrics: |-
                  [Chorus]
                  Modified lyrics for this section
                variety: 1
                vocal_gender: m
                style_weight: 0.65
                weirdness_constraint: 0.65
                audio_weight: 0.65
                persona_id: persona_123
                persona_model: style_persona
      responses:
        '200':
          description: Request successful
          content:
            application/json:
              schema:
                allOf:
                  - type: object
                    properties:
                      code:
                        type: integer
                        enum:
                          - 200
                          - 401
                          - 402
                          - 404
                          - 409
                          - 422
                          - 429
                          - 451
                          - 455
                          - 500
                        description: >-
                          Status code of the request. The HTTP status is 200
                          even when the request fails, so always check this
                          field: 200 means success; any other value means the
                          request failed and msg describes the reason. Common
                          values: 401 invalid or missing API key; 402
                          insufficient credits; 429 rate limit exceeded; 422 or
                          500 the request could not be processed (read msg and
                          check your parameters before retrying).
                      msg:
                        type: string
                        description: Error message when code != 200
                        examples:
                          - success
                    x-apidog-orders:
                      - code
                      - msg
                  - type: object
                    properties:
                      data:
                        type: object
                        properties:
                          taskId:
                            type: string
                            description: >-
                              Task ID for tracking task status. Poll GET
                              /api/v1/jobs/recordInfo?taskId=... (see [Get Task
                              Details](/market/common/get-task-detail)). For
                              Suno audio tasks the tracks are in
                              data.response.data[] (audio_url, stream_audio_url,
                              image_url, id, duration).
                            examples:
                              - 5c79****be8e
                        x-apidog-orders:
                          - taskId
                    x-apidog-orders:
                      - data
          headers: {}
          x-apidog-name: ''
      security:
        - BearerAuth: []
          x-apidog:
            schemeGroups:
              - id: kn8M4YUlc5i0A0179ezwx
                schemeIds:
                  - BearerAuth
            required: true
            use:
              id: kn8M4YUlc5i0A0179ezwx
            scopes:
              kn8M4YUlc5i0A0179ezwx:
                BearerAuth: []
      callbacks:
        audioGenerated:
          '{request.body#/callBackUrl}':
            post:
              description: >-
                When audio generation is complete, the system will call this
                callback to notify the result.


                ### Callback Example

                ```json

                {
                  "code": 200,
                  "msg": "All generated successfully.",
                  "data": {
                    "callbackType": "complete",
                    "task_id": "2fac****9f72",
                    "data": [
                      {
                        "id": "e231****-****-****-****-****8cadc7dc",
                        "audio_url": "https://example.cn/****.mp3",
                        "stream_audio_url": "https://example.cn/****",
                        "image_url": "https://example.cn/****.jpeg",
                        "prompt": "A calm and relaxing piano track.",
                        "model_name": "chirp-v3-5",
                        "title": "Relaxing Piano",
                        "tags": "Jazz",
                        "createTime": 1791200596437,
                        "duration": 198.44
                      }
                    ]
                  }
                }

                ```
              requestBody:
                content:
                  application/json:
                    schema:
                      type: object
                      properties:
                        code:
                          type: integer
                          description: Status code
                          example: 200
                        msg:
                          type: string
                          description: Return message
                          example: All generated successfully
                        data:
                          type: object
                          properties:
                            callbackType:
                              type: string
                              description: >-
                                Callback type: text (text generation complete),
                                first (first song complete), complete (all
                                complete)
                              enum:
                                - text
                                - first
                                - complete
                            task_id:
                              type: string
                              description: Task ID
                            data:
                              type: array
                              items:
                                type: object
                                properties:
                                  id:
                                    type: string
                                    description: Audio unique identifier (audioId)
                                  audio_url:
                                    type: string
                                    description: Audio file URL
                                  stream_audio_url:
                                    type: string
                                    description: Streaming audio URL
                                  image_url:
                                    type: string
                                    description: Cover image URL
                                  prompt:
                                    type: string
                                    description: Generation prompt/lyrics
                                  model_name:
                                    type: string
                                    description: Model name used
                                  title:
                                    type: string
                                    description: Music title
                                  tags:
                                    type: string
                                    description: Music tags
                                  createTime:
                                    type: integer
                                    description: >-
                                      Creation time (Unix timestamp in
                                      milliseconds)
                                  duration:
                                    type: number
                                    description: Audio duration (seconds)
              responses:
                '200':
                  description: Callback received successfully
                  content:
                    application/json:
                      schema:
                        allOf:
                          - type: object
                            properties:
                              code:
                                type: integer
                                enum:
                                  - 200
                                  - 400
                                  - 408
                                  - 413
                                  - 500
                                  - 501
                                  - 531
                                description: >-
                                  Status code. 200 means success; any other
                                  value means failure (for example, 501).
                              msg:
                                type: string
                                description: Error message when code != 200
                                example: success
                      example:
                        code: 200
                        msg: success
      x-apidog-folder: docs/en/Market/Music Models/Suno/Music Generation
      x-apidog-status: released
      x-run-in-apidog: https://app.apidog.com/web/project/1184766/apis/api-43206430-run
components:
  schemas: {}
  securitySchemes:
    BearerAuth:
      type: bearer
      scheme: bearer
      bearerFormat: API Key
      description: >-
        All API requests require a Bearer Token. Add the header `Authorization:
        Bearer YOUR_API_KEY` to authenticate requests.
    BearerAuth1:
      type: bearer
      scheme: bearer
      bearerFormat: API Key
      description: >-
        所有 API 请求都需要 Bearer Token。请在请求头中添加 `Authorization: Bearer YOUR_API_KEY`
        进行身份验证。
servers:
  - url: https://api.kie.ai
    description: 正式环境
security:
  - BearerAuth: []
    x-apidog:
      schemeGroups:
        - id: kn8M4YUlc5i0A0179ezwx
          schemeIds:
            - BearerAuth
      required: true
      use:
        id: kn8M4YUlc5i0A0179ezwx
      scopes:
        kn8M4YUlc5i0A0179ezwx:
          BearerAuth: []

```

# Generate Lyrics

## OpenAPI Specification

```yaml
openapi: 3.0.1
info:
  title: ''
  description: ''
  version: 1.0.0
paths:
  /api/v1/jobs/createTask:
    post:
      summary: Generate Lyrics
      deprecated: false
      description: >
        :::warning Document updated

        If you have already completed the integration process previously, you
        can still access the old version document at: Old version address
        (https://docs.kie.ai/old-model/suno-api/generate-lyrics)

        :::

        Generate creative lyrics content based on a text prompt.


        ### Usage Guide

        - Use this endpoint to create lyrics for music composition

        - Multiple variations of lyrics will be generated for each request

        - Each generated lyric set includes title and structured verse/chorus
        sections


        ### Parameter Details

        - `prompt` should describe the theme, style, or subject of the desired
        lyrics

        - A detailed prompt yields more targeted and relevant lyrics


        ### Developer Notes

        - Generated lyrics are retained for 14 days

        - Callback occurs once with all generated variations when complete

        - Typically returns 2-3 different lyric variations per request

        - Each lyric set is formatted with standard section markers ([Verse],
        [Chorus], etc.)

        - When polling recordInfo, lyrics are in
        resultJson.resultObject.lyricsData[] (text, title, status,
        error_message, tags).
      operationId: generate-lyrics
      tags:
        - docs/en/Market/Music Models/Suno/Lyrics Generation
      parameters: []
      requestBody:
        content:
          application/json:
            schema:
              type: object
              required:
                - model
                - input
              properties:
                model:
                  type: string
                  description: The model name to use for generation. Required field.
                  examples:
                    - ai-music-api/generate-lyrics
                callBackUrl:
                  type: string
                  format: uri
                  description: >-
                    The URL to receive generation task completion updates.
                    Optional but recommended for production use.


                    - System will POST task status and results to this URL when
                    generation completes

                    - Callback includes generated content URLs and task
                    information

                    - Your callback endpoint should accept POST requests with
                    JSON payload containing results

                    - Alternatively, use the Get Task Details endpoint to poll
                    task status

                    - To ensure callback security, see [Webhook Verification
                    Guide](/common-api/webhook-verification) for signature
                    verification implementation
                  examples:
                    - https://api.example.com/callback
                input:
                  type: object
                  description: Input parameters for the generation task
                  properties:
                    prompt:
                      type: string
                      description: >-
                        Description of the desired lyrics content. Be specific
                        about theme, mood, style, or story elements you want in
                        the lyrics. More detailed prompts yield better results.
                        The maximum word limit is 200 characters.
                      examples:
                        - >-
                          A nostalgic song about childhood memories and growing
                          up in a small town
                  required:
                    - prompt
                  x-apidog-orders:
                    - prompt
                  x-apidog-ignore-properties: []
              x-apidog-orders:
                - model
                - callBackUrl
                - input
              x-apidog-ignore-properties: []
            example:
              model: ai-music-api/generate-lyrics
              callBackUrl: https://api.example.com/callback
              input:
                prompt: >-
                  A nostalgic song about childhood memories and growing up in a
                  small town
      responses:
        '200':
          description: Request successful
          content:
            application/json:
              schema:
                allOf:
                  - type: object
                    properties:
                      code:
                        type: integer
                        enum:
                          - 200
                          - 400
                          - 401
                          - 404
                          - 405
                          - 413
                          - 429
                          - 430
                          - 455
                          - 500
                        description: >-
                          Status code of the request. The HTTP status is 200
                          even when the request fails, so always check this
                          field: 200 means success; any other value means the
                          request failed and msg describes the reason. Common
                          values: 401 invalid or missing API key; 402
                          insufficient credits; 429 rate limit exceeded; 422 or
                          500 the request could not be processed (read msg and
                          check your parameters before retrying).
                      msg:
                        type: string
                        description: Error message when code != 200
                        examples:
                          - success
                    x-apidog-orders:
                      - code
                      - msg
                    x-apidog-ignore-properties: []
                  - type: object
                    properties:
                      data:
                        type: object
                        properties:
                          taskId:
                            type: string
                            description: Task ID for tracking task status
                            examples:
                              - 5c79****be8e
                        x-apidog-orders:
                          - taskId
                        x-apidog-ignore-properties: []
                    x-apidog-orders:
                      - data
                    x-apidog-ignore-properties: []
          headers: {}
          x-apidog-name: ''
        '500':
          description: Server error.
          content:
            application/json:
              schema:
                type: object
                properties:
                  code:
                    type: integer
                    description: >-
                      Response status code


                      - **200**: Success - Request has been processed
                      successfully

                      - **401**: Unauthorized - Authentication credentials are
                      missing or invalid

                      - **402**: Insufficient Credits - Account does not have
                      enough credits to perform the operation

                      - **404**: Not Found - The requested resource or endpoint
                      does not exist

                      - **408**: Timeout - No result has been returned for over
                      10 minutes.

                      - **422**: Validation Error - The request parameters
                      failed validation checks

                      - **429**: Rate Limited - Request limit has been exceeded
                      for this resource

                      - **455**: Service Unavailable - System is currently
                      undergoing maintenance

                      - **500**: Server Error - An unexpected error occurred
                      while processing the request

                      - **501**: Generation Failed - Content generation task
                      failed

                      - **505**: Feature Disabled - The requested feature is
                      currently disabled
                  msg:
                    type: string
                    description: Response message, error description when failed
                  data:
                    type: object
                    properties: {}
                    x-apidog-orders: []
                    x-apidog-ignore-properties: []
                x-apidog-orders:
                  - code
                  - msg
                  - data
                required:
                  - code
                  - msg
                  - data
                x-apidog-ignore-properties: []
              example:
                code: 500
                msg: >-
                  Server Error - An unexpected error occurred while processing
                  the request
                data: null
          headers: {}
          x-apidog-name: Error
      security:
        - BearerAuth: []
          x-apidog:
            schemeGroups:
              - id: kn8M4YUlc5i0A0179ezwx
                schemeIds:
                  - BearerAuth
            required: true
            use:
              id: kn8M4YUlc5i0A0179ezwx
            scopes:
              kn8M4YUlc5i0A0179ezwx:
                BearerAuth: []
      callbacks:
        audioLyricsGenerated:
          '{$request.body#/callBackUrl}':
            post:
              description: >-
                System will call this callback when lyrics generation is
                complete.


                ### Callback Example

                ```json

                {
                  "code": 200,
                  "msg": "All generated successfully.",
                  "data": {
                    "callbackType": "complete",
                    "task_id": "3b66882fde0a5d398bd269cab6d9542b",
                    "data": [
                      {
                        "error_message": "",
                        "status": "complete",
                        "text": "[Verse]\nMoonlight spreads across the windowsill\nStars dance, never standing still\nNight breeze weaves dreams with gentle skill\nLeaving all worries on the hill\n\n[Verse 2]\nLights reflect in your bright eyes\nLike meteors across the skies\nThe world stops in that moment's prize\nChasing future, no goodbyes\n\n[Chorus]\nIn starry dreams we find tomorrow\nBreak free from ordinary sorrow\nAll our dreams will bloom and follow\nDon't fear the path, don't fear tomorrow",
                        "title": "Starry Night Dreams"
                      },
                      {
                        "error_message": "",
                        "status": "complete",
                        "text": "[Verse]\nMoonlight spreads across the windowsill\nStars dance, never standing still\nNight breeze weaves dreams with gentle skill\nLeaving all worries on the hill\n\n[Verse 2]\nLights reflect in your bright eyes\nLike meteors across the skies\nThe world stops in that moment's prize\nChasing future, no goodbyes\n\n[Chorus]\nIn starry dreams we find tomorrow\nBreak free from ordinary sorrow\nAll our dreams will bloom and follow\nDon't fear the path, don't fear tomorrow",
                        "title": "Main Street Summer"
                      }
                    ]
                  }
                }

                ```
              requestBody:
                content:
                  application/json:
                    schema:
                      allOf:
                        - type: object
                          properties:
                            code:
                              type: integer
                              enum:
                                - 200
                                - 400
                                - 500
                              description: >-
                                Response status code


                                - **200**: Success - Request has been processed
                                successfully

                                - **400**: Please try rephrasing   with more
                                specific details or using a different approach.

                                Song Description contained artist name

                                Song Description flagged for moderation

                                Unable to generate lyrics from song  
                                description

                                - **500**: Internal Error - Please try again
                                later.
                            msg:
                              type: string
                              description: Error message when code != 200
                              example: success
                        - type: object
                          properties:
                            data:
                              type: object
                              properties:
                                taskId:
                                  type: string
                                  description: Task ID for tracking task status
                                  example: 5c79****be8e
                      type: object
                      properties:
                        code:
                          type: integer
                          description: Status code
                          example: 200
                        msg:
                          type: string
                          description: Response message
                          example: All generated successfully
                        data:
                          type: object
                          properties:
                            callbackType:
                              type: string
                              description: Callback type, fixed as complete
                              enum:
                                - complete
                              example: complete
                            task_id:
                              type: string
                              description: Task ID
                            data:
                              type: array
                              description: Generated lyrics list
                              items:
                                type: object
                                properties:
                                  text:
                                    type: string
                                    description: Lyrics content
                                  title:
                                    type: string
                                    description: Lyrics title
                                  status:
                                    type: string
                                    description: Generation status
                                    enum:
                                      - complete
                                      - failed
                                  error_message:
                                    type: string
                                    description: Error message, valid when status is failed
              responses:
                '200':
                  description: Callback received successfully
      x-apidog-folder: docs/en/Market/Music Models/Suno/Lyrics Generation
      x-apidog-status: released
      x-run-in-apidog: https://app.apidog.com/web/project/1184766/apis/api-43206404-run
components:
  schemas: {}
  securitySchemes:
    BearerAuth:
      type: bearer
      scheme: bearer
      bearerFormat: API Key
      description: >-
        All API requests require a Bearer Token. Add the header `Authorization:
        Bearer YOUR_API_KEY` to authenticate requests.
    BearerAuth1:
      type: bearer
      scheme: bearer
      bearerFormat: API Key
      description: >-
        所有 API 请求都需要 Bearer Token。请在请求头中添加 `Authorization: Bearer YOUR_API_KEY`
        进行身份验证。
servers:
  - url: https://api.kie.ai
    description: 正式环境
security:
  - BearerAuth: []
    x-apidog:
      schemeGroups:
        - id: kn8M4YUlc5i0A0179ezwx
          schemeIds:
            - BearerAuth
      required: true
      use:
        id: kn8M4YUlc5i0A0179ezwx
      scopes:
        kn8M4YUlc5i0A0179ezwx:
          BearerAuth: []

```


# Generate Music Cover

## OpenAPI Specification

```yaml
openapi: 3.0.1
info:
  title: ''
  description: ''
  version: 1.0.0
paths:
  /api/v1/jobs/createTask:
    post:
      summary: Generate Music Cover
      deprecated: false
      description: >-
        :::warning Document updated

        If you have already completed the integration process previously, you
        can still access the old version document at: Old version address
        (https://docs.kie.ai/old-model/suno-api/cover-suno)

        :::

        Generate personalized cover images based on original music tasks.


        ### Usage Guide

        - Use this interface to create personalized cover images for generated
        music

        - Requires the task_id of the original music task

        - Results will be notified through the callBackUrl URL upon completion


        ### Parameter Details

        - `task_id` identifies the unique identifier of the original music
        generation task

        - `callBackUrl` receives callBackUrl address for completion
        notifications


        ### Developer Notes

        - Cover image file URLs will be retained for 14 days

        - It's recommended to call this interface after music generation is
        complete

        - Usually generates 2 different style images for selection
      operationId: generate-cover
      tags:
        - docs/en/Market/Music Models/Suno/Music Generation
      parameters: []
      requestBody:
        content:
          application/json:
            schema:
              type: object
              required:
                - model
                - input
              properties:
                model:
                  type: string
                  description: The model name to use for generation. Required field.
                  examples:
                    - ai-music-api/cover-generate
                callBackUrl:
                  type: string
                  format: uri
                  description: >-
                    The URL to receive generation task completion updates.
                    Optional but recommended for production use.


                    - System will POST task status and results to this URL when
                    generation completes

                    - Callback includes generated content URLs and task
                    information

                    - Your callback endpoint should accept POST requests with
                    JSON payload containing results

                    - Alternatively, poll GET /api/v1/jobs/recordInfo?taskId=...
                    (see [Get Task Details](/market/common/get-task-detail)).

                    - To ensure callback security, see [Webhook Verification
                    Guide](/common-api/webhook-verification) for signature
                    verification implementation
                  examples:
                    - https://api.example.com/callback
                input:
                  type: object
                  description: Input parameters for the generation task
                  properties:
                    task_id:
                      type: string
                      description: >-
                        Original music task ID, should be the task_id returned
                        by the music generation interface.
                      examples:
                        - 73d6128b3523a0079df10da9471017c8
                  required:
                    - task_id
                  x-apidog-orders:
                    - task_id
                  x-apidog-ignore-properties: []
              x-apidog-orders:
                - model
                - callBackUrl
                - input
              x-apidog-ignore-properties: []
            example:
              model: ai-music-api/cover-generate
              callBackUrl: https://api.example.com/callback
              input:
                task_id: <taskId from Generate Music>
      responses:
        '200':
          description: Success
          content:
            application/json:
              schema:
                allOf:
                  - type: object
                    properties:
                      code:
                        type: integer
                        enum:
                          - 200
                          - 400
                          - 401
                          - 402
                          - 404
                          - 409
                          - 422
                          - 429
                          - 455
                          - 500
                        description: >-
                          Status code of the request. The HTTP status is 200
                          even when the request fails, so always check this
                          field: 200 means success; any other value means the
                          request failed and msg describes the reason. Common
                          values: 401 invalid or missing API key; 402
                          insufficient credits; 429 rate limit exceeded; 422 or
                          500 the request could not be processed (read msg and
                          check your parameters before retrying).
                      msg:
                        type: string
                        description: Error message when code != 200
                        examples:
                          - success
                    x-apidog-orders:
                      - code
                      - msg
                    x-apidog-ignore-properties: []
                type: object
                properties:
                  code:
                    type: integer
                    format: int32
                    description: >-
                      Status code of the request. The HTTP status is 200 even
                      when the request fails, so always check this field: 200
                      means success; any other value means the request failed
                      and msg describes the reason. Common values: 401 invalid
                      or missing API key; 402 insufficient credits; 429 rate
                      limit exceeded; 422 or 500 the request could not be
                      processed (read msg and check your parameters before
                      retrying).
                    examples:
                      - 200
                  msg:
                    type: string
                    description: Status message
                    examples:
                      - success
                  data:
                    type: object
                    properties:
                      taskId:
                        type: string
                        description: Task ID
                        examples:
                          - 21aee3c3c2a01fa5e030b3799fa4dd56
                    x-apidog-orders:
                      - taskId
                    x-apidog-ignore-properties: []
                x-apidog-orders:
                  - code
                  - msg
                  - data
                x-apidog-ignore-properties: []
              example:
                code: 200
                msg: success
                data:
                  taskId: 21aee3c3c2a01fa5e030b3799fa4dd56
          headers: {}
          x-apidog-name: ''
        '500':
          description: Server error.
          content:
            application/json:
              schema:
                type: object
                properties:
                  code:
                    type: integer
                    description: >-
                      Response status code


                      - **200**: Success - Request has been processed
                      successfully

                      - **401**: Unauthorized - Authentication credentials are
                      missing or invalid

                      - **402**: Insufficient Credits - Account does not have
                      enough credits to perform the operation

                      - **404**: Not Found - The requested resource or endpoint
                      does not exist

                      - **408**: Timeout - No result has been returned for over
                      10 minutes.

                      - **422**: Validation Error - The request parameters
                      failed validation checks

                      - **429**: Rate Limited - Request limit has been exceeded
                      for this resource

                      - **455**: Service Unavailable - System is currently
                      undergoing maintenance

                      - **500**: Server Error - An unexpected error occurred
                      while processing the request

                      - **501**: Generation Failed - Content generation task
                      failed

                      - **505**: Feature Disabled - The requested feature is
                      currently disabled
                  msg:
                    type: string
                    description: Response message, error description when failed
                  data:
                    type: object
                    properties: {}
                    x-apidog-orders: []
                    x-apidog-ignore-properties: []
                x-apidog-orders:
                  - code
                  - msg
                  - data
                required:
                  - code
                  - msg
                  - data
                x-apidog-ignore-properties: []
              example:
                code: 500
                msg: >-
                  Server Error - An unexpected error occurred while processing
                  the request
                data: null
          headers: {}
          x-apidog-name: Error
      security:
        - BearerAuth: []
          x-apidog:
            schemeGroups:
              - id: kn8M4YUlc5i0A0179ezwx
                schemeIds:
                  - BearerAuth
            required: true
            use:
              id: kn8M4YUlc5i0A0179ezwx
            scopes:
              kn8M4YUlc5i0A0179ezwx:
                BearerAuth: []
      callbacks:
        onCoverGenerated:
          '{$request.body#/callBackUrl}':
            post:
              summary: Cover generation completion callback
              description: >-
                When Cover generation is complete, the system will send a POST
                request to the provided callback URL to notify results
              requestBody:
                required: true
                content:
                  application/json:
                    schema:
                      allOf:
                        - type: object
                          properties:
                            code:
                              type: integer
                              enum:
                                - 200
                                - 500
                              description: >-
                                Response status code


                                - **200**: Success - Request processed
                                successfully

                                - **500**: Internal error - Please try again
                                later.
                            msg:
                              type: string
                              description: Error message when code != 200
                              example: success
                      type: object
                      required:
                        - code
                        - msg
                        - data
                      properties:
                        code:
                          type: integer
                          description: Status code, 200 indicates success
                          example: 200
                        msg:
                          type: string
                          description: Status message
                          example: success
                        data:
                          type: object
                          required:
                            - taskId
                            - images
                          properties:
                            taskId:
                              type: string
                              description: Unique identifier of the generation task
                              example: 21aee3c3c2a01fa5e030b3799fa4dd56
                            images:
                              type: array
                              items:
                                type: string
                              description: >-
                                Array of accessible cover image URLs, valid for
                                14 days
                              example:
                                - >-
                                  https://tempfile.aiquickdraw.com/s/1753958521_6c1b3015141849d1a9bf17b738ce9347.png
                                - >-
                                  https://tempfile.aiquickdraw.com/s/1753958524_c153143acc6340908431cf0e90cbce9e.png
              responses:
                '200':
                  description: Callback received successfully
      x-apidog-folder: docs/en/Market/Music Models/Suno/Music Generation
      x-apidog-status: released
      x-run-in-apidog: https://app.apidog.com/web/project/1184766/apis/api-43206427-run
components:
  schemas: {}
  securitySchemes:
    BearerAuth:
      type: bearer
      scheme: bearer
      bearerFormat: API Key
      description: >-
        All API requests require a Bearer Token. Add the header `Authorization:
        Bearer YOUR_API_KEY` to authenticate requests.
    BearerAuth1:
      type: bearer
      scheme: bearer
      bearerFormat: API Key
      description: >-
        所有 API 请求都需要 Bearer Token。请在请求头中添加 `Authorization: Bearer YOUR_API_KEY`
        进行身份验证。
servers:
  - url: https://api.kie.ai
    description: 正式环境
security:
  - BearerAuth: []
    x-apidog:
      schemeGroups:
        - id: kn8M4YUlc5i0A0179ezwx
          schemeIds:
            - BearerAuth
      required: true
      use:
        id: kn8M4YUlc5i0A0179ezwx
      scopes:
        kn8M4YUlc5i0A0179ezwx:
          BearerAuth: []

```


# Vocal & Instrument Stem Separation

## OpenAPI Specification

```yaml
openapi: 3.0.1
info:
  title: ''
  description: ''
  version: 1.0.0
paths:
  /api/v1/jobs/createTask:
    post:
      summary: Vocal & Instrument Stem Separation
      deprecated: false
      description: >-
        :::warning Document updated

        If you have already completed the integration process previously, you
        can still access the old version document at: Old version address
        (https://docs.kie.ai/old-model/suno-api/separate-vocals)

        :::

        Use advanced audio processing technology to separate music into vocals,
        accompaniment, and individual instrumental stems.


        ### Usage Guide


        - Separate platform-generated mixes into vocals, accompaniment, and
        individual instrumental components.

        - Supports three processing modes:
          - separate_vocal — 2-stem separation (vocals + accompaniment)
          - split_stem — Up to 12-stem separation
          - split_stem_advanced — Advanced multi-stem separation, supporting precise extraction of a single target stem through `stem_name`
        - Suitable for karaoke production, mixing, sample extraction, and
        detailed post-production.

        - Works best with professional AI-generated mixes that have clearly
        separated vocals and instruments.

        - **Billing:** Each API call consumes credits; **repeated calls for the
        same audio stem will be charged again** (no server-side caching).

        - **Pricing:** Billed per request; the price depends on the separation
        mode (`type`). Failed tasks are not charged. For current prices, see
        [kie.ai/pricing](https://kie.ai/pricing).


        ### Separation Mode Description


        | **Mode (`type`)** | **Returned Stems** | **Typical Use Cases** |

        | :---------------------------- |
        :---------------------------------------------------------------------------------------------------------------------------------
        | :------------------------------------------ |

        | `separate_vocal` *(default)*   | **2 stems** — Vocals +
        accompaniment                                                                                                             
        | Quick vocal removal, karaoke, and basic mixing               |

        | `split_stem`                  | **Up to 12 stems** — Vocals, backing
        vocals, drums, bass, guitar, keyboard, strings, brass, woodwinds,
        percussion, synthesizer, FX/other;    | Advanced mixing, remixing, sound
        design, and individual instrument extraction         |

        | `split_stem_advanced`    | Returns 2 groups, each with "extract" (only
        the target stem) and "remove" (everything except the target stem). |
        Individual instrument extraction and detailed post-production      |


        ### Parameters


        | Name       | Type   |
        Description                                                                
        |

        | :--------- | :----- |
        :------------------------------------------------------------------- |

        | `task_id`   | string | The ID of the original music generation
        task                                                |

        | `audio_url`   | string | The URL of the user-uploaded audio file.
        Required when processing user-uploaded audio. Cannot be used together
        with `audio_id`.                                               |

        | `audio_id`  | string | Specifies the audio variant to process when
        multiple versions exist. Cannot be used together with
        `audio_url`.                                 |

        | `type`     | string | Separation mode: `separate_vocal`, `split_stem`,
        or `split_stem_advanced`; defaults to `separate_vocal`     |

        | `stem_name` | string | Used only with type split_stem_advanced, where
        it is required. |


        ### Query Results


        When polling GET /api/v1/jobs/recordInfo, results are in
        resultJson.resultUrls (file names end with the stem name, e.g.
        _Vocals.mp3). Stem ids (needed by Generate MIDI and Convert to WAV
        Format) are returned in the callback
        (vocal_removal_info.origin_data[].id) and in
        resultJson.resultObject.origin_data[].id when polling recordInfo.


        ### Developer Notes


        - All returned audio file URLs are valid for **14 days**.

        - Separation quality depends on the complexity and mixing
        characteristics of the original track.

        - `separate_vocal` returns **2 stems** — vocals + accompaniment.

        - `split_stem` returns **up to 12 independent stems** — vocals, backing
        vocals, drums, bass, guitar, keyboard, strings, brass, woodwinds,
        percussion, synthesizer, and FX/other.

        - `split_stem_advanced` supports specifying a single target stem through
        `stem_name` for precise extraction.

        - Each separated stem can be converted to WAV: call Convert to WAV
        Format with this task's task_id and the stem's id as audio_id.

        - **Billing:** Each request consumes credits. Repeatedly submitting the
        same stem will **deduct credits again** (no server-side caching).
      operationId: separate-vocals
      tags:
        - docs/en/Market/Music Models/Suno/Vocal Removal
      parameters: []
      requestBody:
        content:
          application/json:
            schema:
              type: object
              required:
                - model
                - input
              properties:
                model:
                  type: string
                  description: The model name to use for generation. Required field.
                  examples:
                    - ai-music-api/separate-vocals
                callBackUrl:
                  type: string
                  format: uri
                  description: >-
                    The URL to receive generation task completion updates.
                    Optional but recommended for production use.


                    - System will POST task status and results to this URL when
                    generation completes

                    - Callback includes generated content URLs and task
                    information

                    - Your callback endpoint should accept POST requests with
                    JSON payload containing results

                    - Alternatively, use the Get Task Details endpoint to poll
                    task status

                    - To ensure callback security, see [Webhook Verification
                    Guide](/common-api/webhook-verification) for signature
                    verification implementation
                  examples:
                    - https://api.example.com/callback
                input:
                  type: object
                  description: >-
                    Input parameters for the generation task. Provide either
                    task_id + audio_id, or audio_url alone.
                  oneOf:
                    - title: Separate using existing audio
                      required:
                        - task_id
                        - audio_id
                      properties:
                        task_id:
                          type: string
                          description: >-
                            Unique identifier of the music generation task. This
                            should be a task_id returned from the Generate Music
                            or Extend Music endpoint.
                          examples:
                            - 5c79****be8e
                        audio_id:
                          type: string
                          description: >-
                            Unique identifier of the specific audio track to
                            process for vocal separation. This ID is returned in
                            the callBackUrl data after music generation
                            completes.
                          examples:
                            - e231****-****-****-****-****8cadc7dc
                      not:
                        anyOf:
                          - required:
                              - audio_url
                      x-apidog-orders:
                        - task_id
                        - audio_id
                      x-apidog-ignore-properties: []
                      type: object
                    - title: Separate using user-uploaded audio
                      required:
                        - audio_url
                      properties:
                        audio_url:
                          type: string
                          format: uri
                          description: >-
                            Custom audio URL uploaded by the user. The maximum
                            file size is 20MB.
                          examples:
                            - https://example.com/audio.mp3
                        audio_id:
                          type: string
                          description: >-
                            Unique identifier of the specific audio track to
                            process for vocal separation. This ID is returned in
                            the callBackUrl data after music generation
                            completes. `audio_url` and `audio_id` are mutually
                            exclusive; exactly one of them must be provided.
                          examples:
                            - e231****-****-****-****-****8cadc7dc
                      not:
                        anyOf:
                          - required:
                              - task_id
                          - required:
                              - audio_id
                      x-apidog-orders:
                        - audio_url
                        - audio_id
                      x-apidog-ignore-properties: []
                      type: object
                  properties:
                    type:
                      type: string
                      enum:
                        - separate_vocal
                        - split_stem
                        - split_stem_advanced
                      default: separate_vocal
                      description: >-
                        Separation type, options:


                        - **separate_vocal**: Separate vocals and accompaniment,
                        generating a vocal track and an instrumental track.

                        - **split_stem**: Separate various instrument sounds,
                        generating multiple instrument tracks.

                        - **split_stem_advanced**: Advanced multi-track
                        separation, supporting the specification of a particular
                        instrument track to generate finer instrument tracks.
                      examples:
                        - separate_vocal
                    stem_name:
                      type: string
                      enum:
                        - Lead Vocal
                        - Drum Kit
                        - Kick
                        - Snare
                        - Risers
                        - Bass
                        - Backing Vocals
                        - Piano
                        - Electric Guitar
                        - Percussion
                        - String Section
                        - Synth
                        - Acoustic Guitar
                        - Sound Effects
                        - Synth Pad
                        - Synth Bass
                        - Guitar
                        - Brass Section
                        - Organ
                        - Electronic Drum Kit
                        - Lead Electric Guitar
                        - Synth Keys
                        - Rhythm Electric Guitar
                        - Electric Piano
                        - Upright Bass
                        - Keyboards
                        - Distorted Electric Guitar
                        - Synth Strings
                        - Synth Lead
                        - Woodwinds
                        - Rhythm Acoustic Guitar
                        - Flute
                        - Harp
                        - Tambourine
                        - Trumpet
                        - Arpeggiator
                        - Accordion
                        - Fiddle
                        - Pedal Steel Guitar
                        - Synth Voice
                        - Violin
                        - Digital Piano
                        - Synth Brass
                        - Mandolin
                        - Choir
                        - Banjo
                        - Bells
                        - Clarinet
                        - Tenor Saxophone
                        - Trombone
                        - Shaker
                        - French Horn
                        - Glockenspiel
                        - Electric Bass
                        - Cello
                        - Timpani
                        - Harmonica
                        - Marimba
                        - Vibraphone
                        - Lap Steel Guitar
                        - Saxophone
                        - Orchestra
                        - Horns
                        - Cymbals
                        - Hand Clap
                        - Oboe
                        - Celesta
                        - Congas
                        - Drone
                        - Alto Saxophone
                        - Double Bass
                        - Ukulele
                        - Harpsichord
                        - Baritone Saxophone
                        - Xylophone
                        - Tuba
                        - Bass Guitar
                        - Whistle
                        - Lead Guitar
                        - Rhodes
                        - '808'
                        - Bongos
                        - Bassoon
                        - Cowbell
                        - Viola
                        - Sitar
                        - Steel Drums
                        - Piccolo
                        - Theremin
                        - Bagpipes
                        - Hi-Hat
                        - Music Box
                        - Melodica
                        - Tabla
                        - Koto
                        - Djembe
                        - Taiko
                        - Didgeridoo
                      description: >-
                        Only used when `type` is `split_stem_advanced` to
                        request a specific stem track/instrument name.
                      examples:
                        - Piano
                  x-apidog-orders:
                    - type
                    - stem_name
                  x-apidog-ignore-properties: []
              x-apidog-orders:
                - model
                - callBackUrl
                - input
              x-apidog-ignore-properties: []
            example:
              model: ai-music-api/separate-vocals
              callBackUrl: https://api.example.com/callback
              input:
                task_id: 5c79****be8e
                audio_id: e231****-****-****-****-****8cadc7dc
                type: separate_vocal
      responses:
        '200':
          description: Request successful
          content:
            application/json:
              schema:
                allOf:
                  - type: object
                    properties:
                      code:
                        type: integer
                        enum:
                          - 200
                          - 400
                          - 401
                          - 402
                          - 404
                          - 409
                          - 422
                          - 429
                          - 455
                          - 500
                        description: >-
                          Status code of the request. The HTTP status is 200
                          even when the request fails, so always check this
                          field: 200 means success; any other value means the
                          request failed and msg describes the reason. Common
                          values: 401 invalid or missing API key; 402
                          insufficient credits; 429 rate limit exceeded; 422 or
                          500 the request could not be processed (read msg and
                          check your parameters before retrying).
                      msg:
                        type: string
                        description: Error message when code != 200
                        examples:
                          - success
                    x-apidog-orders:
                      - code
                      - msg
                    x-apidog-ignore-properties: []
                  - type: object
                    properties:
                      data:
                        type: object
                        properties:
                          taskId:
                            type: string
                            description: Task ID for tracking task status
                            examples:
                              - 5c79****be8e
                        x-apidog-orders:
                          - taskId
                        x-apidog-ignore-properties: []
                    x-apidog-orders:
                      - data
                    x-apidog-ignore-properties: []
          headers: {}
          x-apidog-name: ''
        '500':
          description: Server error.
          content:
            application/json:
              schema:
                type: object
                properties:
                  code:
                    type: integer
                    description: >-
                      Response status code


                      - **200**: Success - Request has been processed
                      successfully

                      - **401**: Unauthorized - Authentication credentials are
                      missing or invalid

                      - **402**: Insufficient Credits - Account does not have
                      enough credits to perform the operation

                      - **404**: Not Found - The requested resource or endpoint
                      does not exist

                      - **408**: Timeout - No result has been returned for over
                      10 minutes.

                      - **422**: Validation Error - The request parameters
                      failed validation checks

                      - **429**: Rate Limited - Request limit has been exceeded
                      for this resource

                      - **455**: Service Unavailable - System is currently
                      undergoing maintenance

                      - **500**: Server Error - An unexpected error occurred
                      while processing the request

                      - **501**: Generation Failed - Content generation task
                      failed

                      - **505**: Feature Disabled - The requested feature is
                      currently disabled
                  msg:
                    type: string
                    description: Response message, error description when failed
                  data:
                    type: object
                    properties: {}
                    x-apidog-orders: []
                    x-apidog-ignore-properties: []
                x-apidog-orders:
                  - code
                  - msg
                  - data
                required:
                  - code
                  - msg
                  - data
                x-apidog-ignore-properties: []
              example:
                code: 500
                msg: >-
                  Server Error - An unexpected error occurred while processing
                  the request
                data: null
          headers: {}
          x-apidog-name: Error
      security:
        - BearerAuth: []
          x-apidog:
            schemeGroups:
              - id: kn8M4YUlc5i0A0179ezwx
                schemeIds:
                  - BearerAuth
            required: true
            use:
              id: kn8M4YUlc5i0A0179ezwx
            scopes:
              kn8M4YUlc5i0A0179ezwx:
                BearerAuth: []
      callbacks:
        vocalRemovalGenerated:
          '{$request.body#/callBackUrl}':
            post:
              description: >-
                System will call this callback when vocal separation is
                complete. Callback data structure varies based on the type
                parameter specified in the request.


                ### separate_vocal Type Callback Example

                ```json

                {
                  "code": 200,
                  "msg": "vocal separation generated successfully.",
                  "data": {
                    "task_id": "3e63b4cc88d52611159371f6af5571e7",
                    "vocal_removal_info": {
                      "instrumental_url": "https://file.aiquickdraw.com/s/d92a13bf-c6f4-4ade-bb47-f69738435528_Instrumental.mp3",
                      "origin_url": "",
                      "vocal_url": "https://file.aiquickdraw.com/s/3d7021c9-fa8b-4eda-91d1-3b9297ddb172_Vocals.mp3"
                    }
                  }
                }

                ```


                ### split_stem Type Callback Example

                ```json

                {
                  "code": 200,
                  "msg": "vocal separation generated successfully.",
                  "data": {
                    "task_id": "e649edb7abfd759285bd41a47a634b10",
                    "vocal_removal_info": {
                      "origin_url": "",
                      "backing_vocals_url": "https://file.aiquickdraw.com/s/aadc51a3-4c88-4c8e-a4c8-e867c539673d_Backing_Vocals.mp3",
                      "bass_url": "https://file.aiquickdraw.com/s/a3c2da5a-b364-4422-adb5-2692b9c26d33_Bass.mp3",
                      "brass_url": "https://file.aiquickdraw.com/s/334b2d23-0c65-4a04-92c7-22f828afdd44_Brass.mp3",
                      "drums_url": "https://file.aiquickdraw.com/s/ac75c5ea-ac77-4ad2-b7d9-66e140b78e44_Drums.mp3",
                      "fx_url": "https://file.aiquickdraw.com/s/a8822c73-6629-4089-8f2a-d19f41f0007d_FX.mp3",
                      "guitar_url": "https://file.aiquickdraw.com/s/064dd08e-d5d2-4201-9058-c5c40fb695b4_Guitar.mp3",
                      "keyboard_url": "https://file.aiquickdraw.com/s/adc934e0-df7d-45da-8220-1dba160d74e0_Keyboard.mp3",
                      "percussion_url": "https://file.aiquickdraw.com/s/0f70884d-047c-41f1-a6d0-7044618b7dc6_Percussion.mp3",
                      "strings_url": "https://file.aiquickdraw.com/s/49829425-a5b0-424e-857a-75d4c63a426b_Strings.mp3",
                      "synth_url": "https://file.aiquickdraw.com/s/56b2d94a-eb92-4d21-bc43-3460de0c8348_Synth.mp3",
                      "vocal_url": "https://file.aiquickdraw.com/s/07420749-29a2-4054-9b62-e6a6f8b90ccb_Vocals.mp3",
                      "woodwinds_url": "https://file.aiquickdraw.com/s/d81545b1-6f94-4388-9785-1aaa6ecabb02_Woodwinds.mp3"
                    }
                  }
                }

                ```
              requestBody:
                content:
                  application/json:
                    schema:
                      allOf:
                        - type: object
                          properties:
                            code:
                              type: integer
                              enum:
                                - 200
                                - 500
                              description: >-
                                Response status code


                                - **200**: Success - Request has been processed
                                successfully

                                - **500**: Internal Error - Please try again
                                later.
                            msg:
                              type: string
                              description: Error message when code != 200
                              example: success
                        - type: object
                          properties:
                            data:
                              type: object
                              properties:
                                task_id:
                                  type: string
                                  description: Task ID for tracking task status
                                  example: 5c79****be8e
                      type: object
                      properties:
                        code:
                          type: integer
                          description: Status code
                          example: 200
                        msg:
                          type: string
                          description: Response message
                          example: vocal separation generated successfully.
                        data:
                          type: object
                          properties:
                            task_id:
                              type: string
                              description: Task ID
                            vocal_removal_info:
                              type: object
                              description: >-
                                Vocal separation result information, fields vary
                                based on the type parameter in the request
                              properties:
                                origin_url:
                                  type: string
                                  description: Original audio URL
                                vocal_url:
                                  type: string
                                  description: Vocal part audio URL
                                instrumental_url:
                                  type: string
                                  description: >-
                                    Instrumental part audio URL (separate_vocal
                                    type only)
                                backing_vocals_url:
                                  type: string
                                  description: >-
                                    Backing vocals audio URL (split_stem type
                                    only)
                                drums_url:
                                  type: string
                                  description: Drums part audio URL (split_stem type only)
                                bass_url:
                                  type: string
                                  description: Bass part audio URL (split_stem type only)
                                guitar_url:
                                  type: string
                                  description: Guitar part audio URL (split_stem type only)
                                keyboard_url:
                                  type: string
                                  description: >-
                                    Keyboard part audio URL (split_stem type
                                    only)
                                percussion_url:
                                  type: string
                                  description: >-
                                    Percussion part audio URL (split_stem type
                                    only)
                                strings_url:
                                  type: string
                                  description: >-
                                    Strings part audio URL (split_stem type
                                    only)
                                synth_url:
                                  type: string
                                  description: >-
                                    Synthesizer part audio URL (split_stem type
                                    only)
                                fx_url:
                                  type: string
                                  description: >-
                                    Effects part audio URL (split_stem type
                                    only)
                                brass_url:
                                  type: string
                                  description: Brass part audio URL (split_stem type only)
                                woodwinds_url:
                                  type: string
                                  description: >-
                                    Woodwinds part audio URL (split_stem type
                                    only)
              responses:
                '200':
                  description: Callback received successfully
      x-apidog-folder: docs/en/Market/Music Models/Suno/Vocal Removal
      x-apidog-status: released
      x-run-in-apidog: https://app.apidog.com/web/project/1184766/apis/api-43206408-run
components:
  schemas: {}
  securitySchemes:
    BearerAuth:
      type: bearer
      scheme: bearer
      bearerFormat: API Key
      description: >-
        All API requests require a Bearer Token. Add the header `Authorization:
        Bearer YOUR_API_KEY` to authenticate requests.
    BearerAuth1:
      type: bearer
      scheme: bearer
      bearerFormat: API Key
      description: >-
        所有 API 请求都需要 Bearer Token。请在请求头中添加 `Authorization: Bearer YOUR_API_KEY`
        进行身份验证。
servers:
  - url: https://api.kie.ai
    description: 正式环境
security:
  - BearerAuth: []
    x-apidog:
      schemeGroups:
        - id: kn8M4YUlc5i0A0179ezwx
          schemeIds:
            - BearerAuth
      required: true
      use:
        id: kn8M4YUlc5i0A0179ezwx
      scopes:
        kn8M4YUlc5i0A0179ezwx:
          BearerAuth: []

```

# Create Music Video

## OpenAPI Specification

```yaml
openapi: 3.0.1
info:
  title: ''
  description: ''
  version: 1.0.0
paths:
  /api/v1/jobs/createTask:
    post:
      summary: Create Music Video
      deprecated: false
      description: >
        :::warning Document updated

        If you have already completed the integration process previously, you
        can still access the old version document at: Old version address
        (https://docs.kie.ai/old-model/suno-api/create-music-video)

        :::

        Create a video with visualizations based on your generated music track.


        ### Usage Guide

        - Use this endpoint to turn your audio tracks into visually appealing
        videos

        - Add artist attribution and branding to your music videos

        - Videos can be shared on social media or embedded in websites


        ### Parameter Details

        - `task_id` identifies the original music generation task

        - `audio_id` specifies which audio track to visualize when multiple
        variations exist

        - Optional `author` and `domain_name` add customized branding to the
        video


        ### Developer Notes

        - Generated video files are retained for 14 days

        - Videos are optimized for social media sharing

        - Processing time varies based on audio length and server load

        - When polling GET /api/v1/jobs/recordInfo, the file URL is
        resultJson.resultUrls[0] (the callback uses audio_wav_url / video_url).
      operationId: create-music-video
      tags:
        - docs/en/Market/Music Models/Suno/Music Video Generation
      parameters: []
      requestBody:
        content:
          application/json:
            schema:
              type: object
              required:
                - model
                - input
              properties:
                model:
                  type: string
                  description: The model name to use for generation. Required field.
                  examples:
                    - ai-music-api/create-music-video
                callBackUrl:
                  type: string
                  format: uri
                  description: >-
                    The URL to receive generation task completion updates.
                    Optional but recommended for production use.


                    - System will POST task status and results to this URL when
                    generation completes

                    - Callback includes generated content URLs and task
                    information

                    - Your callback endpoint should accept POST requests with
                    JSON payload containing results

                    - Alternatively, use the Get Task Details endpoint to poll
                    task status

                    - To ensure callback security, see [Webhook Verification
                    Guide](/common-api/webhook-verification) for signature
                    verification implementation
                  examples:
                    - https://api.example.com/callback
                input:
                  type: object
                  description: Input parameters for the generation task
                  properties:
                    task_id:
                      type: string
                      description: >-
                        Unique identifier of the music generation task. This
                        should be a task_id returned from either the "Generate
                        Music" or "Extend Music" endpoints.
                      examples:
                        - taskId_774b9aa0422f
                    audio_id:
                      type: string
                      description: >-
                        Unique identifier of the specific audio track to
                        visualize. This ID is returned in the callBackUrl data
                        after music generation completes.
                      examples:
                        - e231****-****-****-****-****8cadc7dc
                    author:
                      type: string
                      maxLength: 50
                      description: >-
                        Artist or creator name to display as a signature on the
                        video cover. Maximum 50 characters. This creates
                        attribution for the music creator.
                      examples:
                        - DJ Electronic
                    domain_name:
                      type: string
                      maxLength: 50
                      description: >-
                        Website or brand to display as a watermark at the bottom
                        of the video. Maximum 50 characters. Useful for
                        promotional branding or attribution.
                      examples:
                        - music.example.com
                  required:
                    - task_id
                    - audio_id
                  x-apidog-orders:
                    - task_id
                    - audio_id
                    - author
                    - domain_name
                  x-apidog-ignore-properties: []
              x-apidog-orders:
                - model
                - callBackUrl
                - input
              x-apidog-ignore-properties: []
            example:
              model: ai-music-api/create-music-video
              callBackUrl: https://api.example.com/callback
              input:
                task_id: taskId_774b9aa0422f
                audio_id: e231****-****-****-****-****8cadc7dc
                author: DJ Electronic
                domain_name: music.example.com
      responses:
        '200':
          description: Success
          content:
            application/json:
              schema:
                allOf:
                  - type: object
                    properties:
                      code:
                        type: integer
                        enum:
                          - 200
                          - 400
                          - 401
                          - 402
                          - 404
                          - 409
                          - 422
                          - 429
                          - 455
                          - 500
                        description: >-
                          Status code of the request. The HTTP status is 200
                          even when the request fails, so always check this
                          field: 200 means success; any other value means the
                          request failed and msg describes the reason. Common
                          values: 401 invalid or missing API key; 402
                          insufficient credits; 429 rate limit exceeded; 422 or
                          500 the request could not be processed (read msg and
                          check your parameters before retrying).
                      msg:
                        type: string
                        description: Error message when code != 200
                        examples:
                          - success
                    x-apidog-orders:
                      - code
                      - msg
                    x-apidog-ignore-properties: []
                type: object
                properties:
                  code:
                    type: integer
                    format: int32
                    description: >-
                      Status code of the request. The HTTP status is 200 even
                      when the request fails, so always check this field: 200
                      means success; any other value means the request failed
                      and msg describes the reason. Common values: 401 invalid
                      or missing API key; 402 insufficient credits; 429 rate
                      limit exceeded; 422 or 500 the request could not be
                      processed (read msg and check your parameters before
                      retrying).
                    examples:
                      - 200
                  msg:
                    type: string
                    description: Status message
                    examples:
                      - success
                  data:
                    type: object
                    properties:
                      taskId:
                        type: string
                        description: Task ID
                        examples:
                          - ''
                    x-apidog-orders:
                      - taskId
                    x-apidog-ignore-properties: []
                x-apidog-orders:
                  - code
                  - msg
                  - data
                x-apidog-ignore-properties: []
              example:
                code: 200
                msg: success
                data:
                  taskId: <taskId>
          headers: {}
          x-apidog-name: ''
        '500':
          description: Server error.
          content:
            application/json:
              schema:
                type: object
                properties:
                  code:
                    type: integer
                    description: >-
                      Response status code


                      - **200**: Success - Request has been processed
                      successfully

                      - **401**: Unauthorized - Authentication credentials are
                      missing or invalid

                      - **402**: Insufficient Credits - Account does not have
                      enough credits to perform the operation

                      - **404**: Not Found - The requested resource or endpoint
                      does not exist

                      - **408**: Timeout - No result has been returned for over
                      10 minutes.

                      - **422**: Validation Error - The request parameters
                      failed validation checks

                      - **429**: Rate Limited - Request limit has been exceeded
                      for this resource

                      - **455**: Service Unavailable - System is currently
                      undergoing maintenance

                      - **500**: Server Error - An unexpected error occurred
                      while processing the request

                      - **501**: Generation Failed - Content generation task
                      failed

                      - **505**: Feature Disabled - The requested feature is
                      currently disabled
                  msg:
                    type: string
                    description: Response message, error description when failed
                  data:
                    type: object
                    properties: {}
                    x-apidog-orders: []
                    x-apidog-ignore-properties: []
                x-apidog-orders:
                  - code
                  - msg
                  - data
                required:
                  - code
                  - msg
                  - data
                x-apidog-ignore-properties: []
              example:
                code: 500
                msg: >-
                  Server Error - An unexpected error occurred while processing
                  the request
                data: null
          headers: {}
          x-apidog-name: Error
      security:
        - BearerAuth: []
          x-apidog:
            schemeGroups:
              - id: kn8M4YUlc5i0A0179ezwx
                schemeIds:
                  - BearerAuth
            required: true
            use:
              id: kn8M4YUlc5i0A0179ezwx
            scopes:
              kn8M4YUlc5i0A0179ezwx:
                BearerAuth: []
      callbacks:
        onMp4Generated:
          '{$request.body#/callBackUrl}':
            post:
              summary: MP4 Generation Completion Callback
              description: >-
                When MP4 generation is complete, the system will send a POST
                request to the provided callback URL to notify the result
              requestBody:
                required: true
                content:
                  application/json:
                    schema:
                      allOf:
                        - type: object
                          properties:
                            code:
                              type: integer
                              enum:
                                - 200
                                - 500
                              description: >-
                                Response status code


                                - **200**: Success - Request has been processed
                                successfully

                                - **500**: Internal Error - Please try again
                                later.
                            msg:
                              type: string
                              description: Error message when code != 200
                              example: success
                      type: object
                      required:
                        - code
                        - msg
                        - data
                      properties:
                        code:
                          type: integer
                          description: Status code; 200 indicates success.
                          example: 200
                        msg:
                          type: string
                          description: Status message
                          example: msg_9a23a47664f7
                        data:
                          type: object
                          required:
                            - task_id
                            - video_url
                          properties:
                            task_id:
                              type: string
                              description: Unique identifier of the generation task
                              example: task_id_5bbe7721119d
                            video_url:
                              type: string
                              description: Accessible video URL, valid for 14 days
                              example: video_url_847715e66259
              responses:
                '200':
                  description: Callback received successfully
      x-apidog-folder: docs/en/Market/Music Models/Suno/Music Video Generation
      x-apidog-status: released
      x-run-in-apidog: https://app.apidog.com/web/project/1184766/apis/api-43206403-run
components:
  schemas: {}
  securitySchemes:
    BearerAuth:
      type: bearer
      scheme: bearer
      bearerFormat: API Key
      description: >-
        All API requests require a Bearer Token. Add the header `Authorization:
        Bearer YOUR_API_KEY` to authenticate requests.
    BearerAuth1:
      type: bearer
      scheme: bearer
      bearerFormat: API Key
      description: >-
        所有 API 请求都需要 Bearer Token。请在请求头中添加 `Authorization: Bearer YOUR_API_KEY`
        进行身份验证。
servers:
  - url: https://api.kie.ai
    description: 正式环境
security:
  - BearerAuth: []
    x-apidog:
      schemeGroups:
        - id: kn8M4YUlc5i0A0179ezwx
          schemeIds:
            - BearerAuth
      required: true
      use:
        id: kn8M4YUlc5i0A0179ezwx
      scopes:
        kn8M4YUlc5i0A0179ezwx:
          BearerAuth: []

```