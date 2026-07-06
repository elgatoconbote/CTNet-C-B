# CTNet-C-B

CTNet-C-B MAX es una build compacta ejecutable de CTNet orientada a estado latente reactivo.

El eje del repositorio es una arquitectura donde la unidad primaria no es el token, sino el estado empaquetado:

```text
Xi = pack(Z, M, R, C, P)
```

Incluye perfiles de escalado, dinámica reversible, coherencia multivista u/p, observador Cubo6D, memoria procedimental MTHD y operadores verificables.

## Ejecución

```bash
pip install -r requirements.txt
python ctnet_cb_max.py audit --profile smoke
python ctnet_cb_max.py train --profile smoke --steps 2 --batch 2
python ctnet_cb_max.py bench --profile smoke
python ctnet_cb_max.py operator "sum: 2 3 5"
```

## Perfiles

| perfil | Xi | capacidad |
|---|---:|---:|
| smoke | [B,64,16] | 1.024 |
| base | [B,256,64] | 16.384 |
| xl | [B,1024,128] | 131.072 |
| max | [B,4096,256] | 1.048.576 |

## Estado de carga

El repositorio fue inicializado desde ChatGPT mediante el conector oficial de GitHub. La carga directa de archivos grandes/binarios desde el sandbox no está disponible por este conector, así que la build se está subiendo como código fuente UTF-8.
