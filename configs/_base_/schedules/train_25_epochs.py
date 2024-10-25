optim_wrapper=dict(
  optimizer=dict(type='AdamW', lr=1e-3, weight_decay=0.0001),
  clip_grad=dict(max_norm=35, norm_type=2),
)

log_config=dict(
  interval=10,
  hooks=[
    dict(type='TextLoggerHook'),
    dict(type='FileLoggerHook', by_epoch=True, interval=1, out_dir='work_dirs')
  ]
)

hooks=[
  dict(type='CheckpointHook', interval=1)
]