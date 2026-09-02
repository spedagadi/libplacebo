# ML feature extraction tool (requires FFmpeg)
# Standalone binary for extracting 77-dimensional features from video frames
# Used by Python training pipeline to ensure bit-exact feature matching
ffmpeg_deps = []
if cc.has_header('libavformat/avformat.h')
    ffmpeg_deps += dependency('libavformat', version: '>=58.0.0', required: false)
    ffmpeg_deps += dependency('libavcodec', version: '>=58.0.0', required: false)
    ffmpeg_deps += dependency('libavutil', version: '>=56.0.0', required: false)

    if ffmpeg_deps[0].found() and ffmpeg_deps[1].found() and ffmpeg_deps[2].found()
        # pl_extract_features (single-shot) still uses old API — pending rewrite.
        # message('Skipping pl_extract_features single-shot (pending API migration)')

        # Daemon: rewritten to use pl_render_image + pl_renderer_get_ml_features
        pl_extract_features_daemon = executable('pl_extract_features_daemon',
            'pl_extract_features_daemon.c',
            dependencies: [ffmpeg_deps, libplacebo],
            install: true,
            install_dir: get_option('bindir'),
        )
        message('ML feature extraction daemon will be built: pl_extract_features_daemon')
    else
        warning('FFmpeg libraries not found, skipping pl_extract_features')
    endif
else
    warning('FFmpeg headers not found, skipping pl_extract_features')
endif

# DV frame renderer — requires FFmpeg + libplacebo + D3D11
if ffmpeg_deps[0].found() and ffmpeg_deps[1].found() and ffmpeg_deps[2].found()
    d3d11_dep = cc.find_library('d3d11', required: false)
    if d3d11_dep.found()
        dv_render = executable('dv_render',
            'dv_render.c',
            dependencies: [ffmpeg_deps, libplacebo, d3d11_dep],
            install: false,
        )
        message('DV frame renderer will be built: dv_render')
    else
        warning('D3D11 not found, skipping dv_render')
    endif
endif

# libplacebo spline baseline evaluator — no FFmpeg needed, only libplacebo
libplacebo_baseline_eval = executable('libplacebo_baseline_eval',
    'libplacebo_baseline_eval.c',
    dependencies: [libplacebo],
    install: false,
)
message('libplacebo baseline evaluator will be built: libplacebo_baseline_eval')

pl_ml_model_eval = executable('pl_ml_model_eval',
    'pl_ml_model_eval.c',
    dependencies: [libplacebo],
    install: false,
)
message('Native ML model evaluator will be built: pl_ml_model_eval')
