# Experimento sobre la Teoría de la Información en Comunicación Celular

Este proyecto simula la comunicación entre dos células utilizando aspectos clave de la teoría de la información. El objetivo es enviar "mensajes químicos" (moléculas representadas como cadenas de bits) de una célula a otra y usar técnicas de corrección de errores para garantizar que el mensaje se reciba correctamente, incluso si el canal introduce ruido.

## Conceptos de la Teoría de la Información

### Definición

La teoría de la información es un campo interdisciplinario que aborda el almacenamiento, transmisión y procesamiento de información, considerando aspectos matemáticos, estadísticos y computacionales.

### Entropía

La entropía es una medida que cuantifica la incertidumbre o sorpresa asociada con los posibles resultados de una variable aleatoria.

\[
H(X) = -\sum_{x \in X} p(x) \log_2 p(x)
\]

### Información Mutua

La información mutua describe la cantidad de información que una variable aleatoria contiene sobre otra.

\[
I(X; Y) = H(X) + H(Y) - H(X, Y)
\]

### Codificación

La codificación es el proceso de conversión de una fuente de información en una secuencia de símbolos para su transmisión eficiente. Ejemplos comunes son la codificación Huffman y Shannon-Fano.

### Canal de Comunicación

Un canal de comunicación es un medio a través del cual los mensajes se envían de un punto a otro. Puede ser tanto físico como virtual.

### Capacidad de Canal

Es la máxima tasa a la que se puede transmitir información a través de un canal para un nivel de ruido dado, según el teorema de Shannon.

\[
C = B \log_2(1 + \frac{S}{N})
\]

### Ruido y Error

Factores que afectan la precisión de la transmisión. El ruido es cualquier interferencia no deseada.

### Corrección de Errores

Técnicas que detectan y corrigen errores en la transmisión. Ejemplos son los códigos Hamming y Reed-Solomon.

### Teoremas de Shannon

Establecen los límites en la eficiencia de codificación y transmisión para un canal dado. Incluye el teorema de la fuente de codificación y el teorema del canal de codificación.

## El Experimento

### Descripción

Este experimento simula la comunicación entre dos agentes que representan células. Utiliza un enfoque de multiproceso para simular la comunicación en un canal de comunicación ruidoso. Se consideran aspectos como la entropía, la información mutua, la codificación de Huffman y la teoría de Shannon para la capacidad del canal.

### Cómo Ejecutar

Ejecute el script `experiment.py` para realizar el experimento. Se generará una gráfica que muestra cómo el comportamiento de los agentes cambia con el tiempo, junto con la información mutua entre ellos.

## Referencias

- Claude E. Shannon, "A Mathematical Theory of Communication", Bell System Technical Journal, 1948.
- Thomas M. Cover, Joy A. Thomas, "Elements of Information Theory", Wiley-Interscience, 2006.
