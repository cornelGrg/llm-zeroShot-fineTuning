import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.colors import to_rgba, to_hex

def get_custom_colors(models):
    """Genera colori personalizzati per i modelli, con varianti per i modelli addestrati."""
    base_colors = plt.get_cmap('tab10')
    color_map = {}
    assigned_colors = {}
    color_index = 0
    
    sorted_models = sorted(models)
    
    for model in sorted_models:
        base_model = model.replace('_trained', '')
        if base_model not in assigned_colors:
            assigned_colors[base_model] = base_colors(color_index)
            color_index += 1
        
        base_color = assigned_colors[base_model]
        if '_trained' in model:
            # Rendi il colore più scuro per i modelli addestrati
            rgba = to_rgba(base_color)
            darker_color = [max(0, c - 0.2) for c in rgba[:3]]
            color_map[model] = to_hex(darker_color)
        else:
            color_map[model] = to_hex(base_color)
            
    return [color_map[model] for model in sorted_models]

if __name__ == "__main__":
    csv_path = 'accuracy_example_pool_sizes.csv'
    output_dir = 'graphs/matplot'

    # Crea la cartella di output se non esiste
    os.makedirs(output_dir, exist_ok=True)

    # Carica i dati dal file CSV e seleziona le ultime 30 righe
    df = pd.read_csv(csv_path, sep=';').tail(45)

    # Calcola l'accuratezza media raggruppando per 'Test mode' e 'Model'
    mean_accuracy = df.groupby(['Test mode', 'Model'])['Accuracy (%)'].mean().unstack()

    # Definisci l'ordine desiderato per le modalità di test
    test_mode_order = ['zero', 'few', 'def', 'def-few']
    mean_accuracy = mean_accuracy.reindex(test_mode_order)

    # Ottieni i colori personalizzati
    custom_colors = get_custom_colors(mean_accuracy.columns)

    # Crea il grafico a barre
    ax = mean_accuracy.plot(kind='bar', figsize=(12, 8), rot=0, width=0.75, color=custom_colors)

    # Aggiungi etichette e titolo
    plt.title('Confronto Accuratezza Media per Modello e Modalità di Test', fontsize=16)
    plt.ylabel('Accuratezza Media (%)', fontsize=12)
    plt.xlabel('Modalità di Test', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(0, 118) # Imposta il limite dell'asse y per una migliore visualizzazione
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Aggiungi i valori sopra le barre
    for container in ax.containers:
        # Filtra le etichette per i valori non-NaN
        labels = [f'{v:.2f}' if not np.isnan(v) else '' for v in container.datavalues]
        ax.bar_label(container, labels=labels, fontsize=7, padding=3, rotation=0)

    # Migliora il layout e salva il grafico
    plt.legend(title='Modello', loc='upper right', fontsize="8")
    plt.tight_layout() # Aggiusta il layout
    
    output_path = os.path.join(output_dir, 'accuracy_comparison.png')
    plt.savefig(output_path)

    print(f"Grafico salvato in: {output_path}")

    # Mostra il grafico
    plt.show()
