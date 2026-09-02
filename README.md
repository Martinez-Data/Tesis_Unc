# Macroeconomic Nowcasting: IMACEC (YoY %)

Pipeline de nowcasting para anticipar la actividad económica mensual de Chile (IMACEC en variación interanual) antes de su publicación oficial, combinando modelos lineales regularizados, redes neuronales y ensambles bajo un esquema riguroso sin filtración de datos (*data leakage*).

---

## Metodología Clave

* **Validación Out-of-Sample Estricta:** Backtesting con ventana expansiva (*Expanding Window*) reestimado mes a mes en tiempo real (~190 meses evaluados).
* **Modelos Evaluados:**
  * Modelos lineales regularizados con sintonización dinámica de hiperparámetros (`LassoCV`, `RidgeCV`).
  * Red neuronal perceptrón multicapa (`MLPRegressor`) con regularización $L_2$ y *early stopping* para capturar no-linealidades.
  * Ensamble convexo (*Blending* 50/50 Lasso + MLP).
* **Inferencia Estadística:** Test de Diebold-Mariano con corrección por autocorrelación y heterocedasticidad (Newey-West) frente al benchmark ingenuo (*naive*) y a las Expectativas de Mercado.
* **Interpretabilidad:** Selección dinámica de variables a lo largo del tiempo (mapa de calor de coeficientes $L_1$) y *Permutation Feature Importance* sobre la arquitectura neuronal.

---

## Resultados Fuera de Muestra

<!-- Reemplazar con los valores finales de metrics_summary -->
| Modelo | MAE | RMSE | R² | Dir. Accuracy (%) |
| :--- | :---: | :---: | :---: | :---: |
| **LassoCV** | - | - | - | - |
| **RidgeCV** | - | - | - | - |
| **Red_Neuronal_MLP** | - | - | - | - |
| **Ensemble_Lasso_MLP** | - | - | - | - |
| *Mercado (Benchmark)* | - | - | - | - |
| *Naive* | - | - | - | - |

---

## Visualizaciones

### Predicción Fuera de Muestra
![Nowcasting Results](assets/nowcasting_results.png)

### Selección Dinámica de Variables (Lasso Heatmap)
![Evolución de Coeficientes](assets/lasso_coef_evolution.png)

---

## Requisitos y Ejecución

```bash
pip install -r requirements.txt
jupyter notebook notebooks/nowcasting_pipeline.ipynb