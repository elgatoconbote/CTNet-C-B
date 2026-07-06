# CTNet-C-B

CTNet-C-B es una implementación experimental de CTNet como arquitectura neuronal de estado latente reactivo, memoria funcional, dinámica reversible y coherencia en bucle cerrado.

La operación central del sistema no es la predicción autoregresiva de un siguiente token, sino la transformación de un estado latente completo:

Xi(t+1) = Phi_theta(Xi(t), o(t))

donde o(t) puede ser una observación externa, una salida previa, una señal interna, una corrección, una traza del proceso o un resultado operacional.

## Estado latente

CTNet trabaja sobre un estado estructurado empaquetado:

Xi = pack(Z, M, R, C, P)

Z representa la activación principal.
M representa memoria interna y soporte reactivo.
R representa relaciones internas.
C contiene métricas de cierre, error y estabilidad.
P actúa como reserva estructural de grados de libertad.

El tensor global permite operar de forma homogénea, pero sus regiones conservan función diferenciada. El soporte físico puede mantenerse acotado mientras la capacidad funcional crece por modificación de transición, reactividad adquirida, reglas procedimentales y operadores verificables.

## Ciclo operativo

observación -> codificación -> estado empaquetado -> dinámica reversible -> coherencia u/p -> observador interno -> acción visible -> reentrada -> actualización funcional

La salida textual, cuando existe, no es el centro de la arquitectura. Es una acción visible producida por un estado ya transformado. Esa acción vuelve a observarse y entra de nuevo en el sistema como material entrenable.

## Memoria funcional

CTNet separa memoria explícita y memoria funcional.

Memoria explícita: contenido materialmente guardado.
Memoria funcional: cambio estable de respuesta ante estímulos.

Una experiencia queda incorporada cuando modifica la función de respuesta, amplía el atlas de operadores, cambia el núcleo de reactividad o reorganiza la dinámica interna del estado.

## Principio u/p

El principio u/p compara partes y totalidades relativas del estado. Una incoherencia no se interpreta como ruido plano: se convierte en residuo, frontera, dirección de reorganización o necesidad de nuevo operador.

residuo -> energía -> dirección -> operador -> cierre / no cierre

La coherencia estabiliza regiones de cierre. La incoherencia delimita frontera, rechazo y nueva observabilidad.

## Operadores verificables

CTNet-C-B incorpora una capa de resolución verificable. Cuando una familia de problemas puede reducirse a invariantes de dominio, el sistema debe producir una salida positiva mínima y someterla a verificación externa.

problema -> detector -> reducción -> generador -> verificador -> cierre

Si no existe operador verificable aplicable, el sistema no debe fingir cierre: debe señalar necesidad de nuevo operador.

## Instalación

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Auditoría

python ctnet_run_max.py audit --profile smoke
python ctnet_max_audit.py --profile smoke
python tests/test_ctnet_invariants.py

## Entrenamiento mínimo

python ctnet_run_max.py train --profile smoke --synthetic --steps 10 --log-every 1

## Entrenamiento máximo

python ctnet_run_max.py train --profile max --synthetic --steps 100000 --cuda --coherence-grad-scale --save-final checkpoints/ctnet_max_final.pt

## Tesis

CTNet-C-B explora una vía donde aprender no significa únicamente almacenar datos, sino transformar la función de respuesta del sistema. La memoria relevante se expresa como reactividad, cierre interno, operadores verificables y reorganización dinámica del estado.
