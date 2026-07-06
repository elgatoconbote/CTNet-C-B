# CTNet 3.0 MAX FINAL REPORT

## Construcción

Se creó una versión escalada sobre CTNet 2.6 Omega Cubo 6D. El núcleo conserva `pack(Z,M,R,C6,P)`, dinámica reversible, cierre `u/p`, autoobservación, efector textual, MTHD y operadores verificables.

## Perfiles finales

| perfil | Xi | capacidad | semantic_size | pad | parámetros entrenables |
|---|---:|---:|---:|---:|---:|
| smoke | [B,64,16] | 1024 | 797 | 227 | 489 |
| base | [B,256,64] | 16384 | 14365 | 2019 | 4948 |
| xl | [B,1024,128] | 131072 | 122909 | 8163 | 26744 |
| max | [B,4096,256] | 1048576 | 983069 | 65507 | 187584 |

## Validación ejecutada

- `python -m py_compile *.py tests/test_ctnet_invariants.py` pasó.
- `python tests/test_ctnet_invariants.py` pasó.
- `python ctnet_functional_benchmark.py --profile smoke --steps 2 --batch 2` pasó y detectó cambio de reactividad.
- `python ctnet_run_max.py train --profile smoke --synthetic --steps 1 --log-every 1` entrenó offline y guardó checkpoint local de prueba. El checkpoint de prueba fue eliminado del paquete final para dejar la entrega limpia.

## Límite honesto

El paquete contiene arquitectura, escalado, auditoría y ruta de entrenamiento. No contiene un checkpoint grande ya entrenado. El perfil `max` está preparado para GPU grande y debe entrenarse externamente.
