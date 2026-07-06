# CTNet 3.0 MAX FINAL

Esta entrega convierte CTNet 2.6 Omega Cubo 6D en una versión escalable por perfiles, con auditoría de invariantes, benchmark de memoria funcional y capa inicial de operadores verificables.

## Contratos que quedan ejecutables

- Estado latente empaquetado `Xi=[B,N,d]` con `Z`, `M`, `R`, `C6` y `pad`.
- Identidad de capacidad: `semantic_size + pad_size = N*d`.
- Reversibilidad por forward/inverse sobre estado completo.
- MTHD con lectura exacta por recibo y pliegue reversible sobre `M/R`.
- Entrenamiento offline reproducible por flujo sintético cuando no hay red.
- Reentrada de salida y autoobservación conservadas en el trainer original.
- Operadores verificables separados del estado continuo.

## Perfiles

- `smoke`: CPU, auditoría rápida.
- `base`: escala útil para iteración seria.
- `xl`: GPU, capacidad 131072 por muestra.
- `max`: geometría máxima incluida, `Xi=[B,4096,256]`, capacidad 1048576 por muestra.

El perfil `max` no debe ejecutarse en CPU salvo auditorías extremadamente pequeñas. Está preparado para GPU grande.

## Comandos principales

```bash
python ctnet_run_max.py profiles
python ctnet_run_max.py audit --profile smoke --steps 2
python ctnet_run_max.py audit --profile smoke --mthd --steps 1
python ctnet_run_max.py bench --profile smoke --steps 3 --batch 2
python ctnet_run_max.py train --profile smoke --synthetic --steps 10 --log-every 1
```

Entrenamiento máximo:

```bash
bash scripts/train_max_gpu.sh
```

## Lectura estructural

La versión no afirma que el checkpoint ya sea semánticamente competitivo. Lo que fija es la ruta completa de escalado: geometría, cierre `u/p`, reversibilidad, memoria funcional por reactividad, MTHD y operadores verificables. Desde aquí, la mejora real se mide por auditorías y benchmarks, no por declaraciones.
