# CTNet 3.0 MAX FINAL

Versión escalada de CTNet 2.6 Omega Cubo 6D.

## Ejecutar ya

```bash
python tests/test_ctnet_invariants.py
python ctnet_run_max.py profiles
python ctnet_run_max.py train --profile smoke --synthetic --steps 10 --log-every 1
```

## Arquitectura conservada

CTNet sigue siendo un sistema de estado latente reactivo:

```text
Observador -> Enc -> pack(Z,M,R,C6,P) -> Phi reversible -> acción -> reentrada -> actualización
```

La salida textual no es el centro. Es una acción visible que vuelve a entrar como observación. La memoria principal no se mide por slots, sino por cambio estable de respuesta.

## Escalado

| perfil | Xi | capacidad por muestra | parámetros | uso |
|---|---:|---:|---:|---|
| smoke | [B,64,16] | 1024 | 489 | test local |
| base | [B,256,64] | 16384 | 4948 | iteración seria |
| xl | [B,1024,128] | 131072 | 26744 | GPU |
| max | [B,4096,256] | 1048576 | 187584 | GPU grande |

## Lo nuevo en esta versión

- `ctnet_max_profiles.py`: perfiles de geometría escalada.
- `ctnet_run_max.py`: entrada única para auditar, entrenar y benchmarkear.
- `ctnet_max_audit.py`: auditoría de layout, reversibilidad y MTHD.
- `ctnet_functional_benchmark.py`: benchmark de reactividad funcional.
- `ctnet_verifiable_operators.py`: atlas mínimo de operadores verificables.
- `tests/test_ctnet_invariants.py`: contratos ejecutables.
- `train_vram_up_coherence_ctnet.py`: ahora soporta `--synthetic` y `--local-file` para entrenar sin red.

## Comandos máximos

```bash
python ctnet_run_max.py train --profile max --synthetic --steps 100000 --cuda --coherence-grad-scale --save-final checkpoints/ctnet_max_final.pt
```

Para no ejecutar todavía y ver el comando exacto:

```bash
python ctnet_run_max.py train --profile max --synthetic --steps 100000 --cuda --dry-run
```
