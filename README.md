# ProyectoProgra_Paralela
Algoritmo PSO hecho desde 0 sin ayuda de librerías como pyswarm y su utilización en casos de usos y aplicación en un casi real

La idea es que el core del PSO sea el mismo y sólo cambie la estrategia de evaluación/actualización donde aplique. Se pide implementar y comparar al menos estas variantes:

V0. Secuencial (baseline)
•	Evaluación fitness secuencial y actualización secuencial.
V1. Concurrencia con hilos (threading)
•	Paralelizar la evaluación de fitness por partículas con ThreadPoolExecutor.
•	Medir impacto (considerar GIL: justificar cuándo puede/no puede mejorar; por ejemplo si la evaluación llama a NumPy vectorizado o I/O).
V2. Paralelismo con procesos (multiprocessing / ProcessPoolExecutor)
•	Paralelizar la evaluación de fitness por partículas con procesos.
•	Tratar serialización (pickling) y coste de IPC.
•	Incluir “batching” como optimización (enviar bloques de partículas en vez de una por tarea).
V3. Asyncio (concurrencia cooperativa) aplicada con sentido
•	Diseñar un caso de evaluación asimétrica (p. ej., función objetivo que simula latencias o consulta a un servicio local) para que tenga sentido.
•	Implementar evaluación asíncrona (gather) y medir.
V4. Vectorización/numérica (NumPy) como “paralelismo implícito”
•	Implementar evaluación y actualización usando operaciones vectorizadas (matrices) en vez de bucles Python.
•	Comparar contra V0–V3.
V5 (opcional, bonus). Joblib / Ray / Dask / Numba
•	Implementar una variante adicional y justificar el framework elegido.
