

# Proceso Gaussiano Neural Normalizado Espectral (Implementación en PyTorch)
Implementación en PyTorch de SNGP según se encuentra en https://arxiv.org/pdf/2006.10108.pdf.

Este repositorio sigue la implementación encontrada en https://www.tensorflow.org/tutorials/understanding/sngp, pero utiliza PyTorch.
A diferencia del artículo original y esta implementación, el cuaderno también ilustra cómo los principios de SNGP pueden aplicarse a una tarea de regresión para estimar la incertidumbre.

Tenga en cuenta que este proyecto ha sido desarrollado enteramente para uso personal; sin embargo, se distribuye libremente. Se ha puesto a disposición por si puede resultar de utilidad para profesionales e investigadores en ML.

## Desarrollo

Configure un entorno de conda de la siguiente manera:

```bash
micromamba create -f environment.yml
micromamba activate sngp
```

Ejecute el script sngp:

```bash
python sngp.py
```

Genere e inicie el cuaderno Jupyter de sngp:

```bash
jupytext --to ipynb sngp.py
jupyter notebook sngp.ipynb
```

Actualice el archivo markdown de sngp:

```bash
jupyter nbconvert --execute --to markdown sngp.ipynb
```
