# Particle Swarm Optimization (PSO) — Versión Secuencial (V0)

Este proyecto implementa un Particle Swarm Optimization (PSO) canónico en Python, siguiendo principios de ingeniería del software y una arquitectura modular preparada para versiones paralelas y concurrentes posteriores.



## Estructura del proyecto

## core/

Contiene la implementación principal del algoritmo PSO.  
Este módulo no depende de paralelismo, visualización ni persistencia, y no debe modificarse en versiones posteriores.


### particle.py

Define una partícula individual del enjambre.

Qué hace:
- Guarda la posición, la velocidad y la mejor posición histórica de la partícula.
- Actualiza la velocidad y la posición usando la fórmula clásica de PSO.

Cómo lo hace:
- Inicializa posición y velocidad de forma aleatoria.
- Aplica la ecuación:
       v = w·v + c1·r1·(pbest − x) + c2·r2·(gbest − x)
- Actualiza la posición sumando la velocidad.



### swarm.py

Define el enjambre de partículas.

Qué hace:
- Almacena todas las partículas.
- Mantiene la mejor solución global encontrada.

Cómo lo hace:
- Crea una lista de partículas.
- Actualiza el mejor global comparando el fitness de cada partícula.
- Proporciona todas las posiciones del enjambre cuando el PSO las necesita.



### topology.py

Define cómo una partícula obtiene la mejor posición del enjambre.

Qué hace:
- Establece la topología del PSO.

Cómo lo hace:
- Implementa la topología global-best, donde todas las partículas usan la mejor posición global.


### bounds.py

Gestiona los límites del dominio de búsqueda.

Qué hace:
- Evita que las partículas salgan del rango permitido.

Cómo lo hace:
- Usa una estrategia clamp, recortando la posición al intervalo definido.



### evaluator.py

Define la interfaz para evaluar el fitness.

Qué hace:
- Desacopla el PSO de la forma en que se calcula el fitness.

Cómo lo hace:
- Recibe todas las posiciones del enjambre y devuelve un fitness por partícula.
- Permite cambiar la estrategia de evaluación sin modificar el PSO.



### pso.py

Implementa el algoritmo PSO completo.

Qué hace:
- Ejecuta el ciclo principal del PSO.
- Coordina partículas, enjambre, evaluación y actualización.

Cómo lo hace:
- En cada iteración:
1. Obtiene las posiciones del enjambre.
2. Evalúa el fitness.
3. Actualiza los mejores personales y el mejor global.
4. Actualiza velocidades y posiciones.
5. Aplica los límites.
- Guarda el mejor fitness por iteración.




-----------
## objectives/

Contiene funciones objetivo para probar y validar el PSO.



### sphere.py

Define la función objetivo Sphere.

Qué hace:
- Calcula la suma de los cuadrados de un vector.

Cómo lo hace:
- Aplica la fórmula:
    f(x) = Σ xᵢ²

## run_pso.py

Script principal para ejecutar una corrida del PSO.

Qué hace:
- Configura el problema y los parámetros del PSO.
- Ejecuta una simulación completa.
- Muestra la mejor solución encontrada.

Cómo lo hace:
- Usa evaluación secuencial.
- Llama al método run() del PSO.

---



## Estado actual del proyecto

- Implementación secuencial del PSO (V0).
- Arquitectura modular y extensible.
- Preparado para versiones paralelas y concurrentes.
