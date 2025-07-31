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

def mode_comparison_graph(csv_path, output_dir, trained_size_to_show=None):
    # Carica i dati dal file CSV
    df = pd.read_csv(csv_path, sep=';').tail(22)  #regola tail in base agli ultimi test effettuati

    if trained_size_to_show is not None:
        # Filtra per mantenere i modelli base e solo la dimensione specificata dei modelli addestrati
        is_base_model = ~df['Model'].str.contains('_trained_')
        is_specific_trained_model = df['Model'].str.contains(f'_trained_{trained_size_to_show}')
        df = df[is_base_model | is_specific_trained_model].copy()

    # Calcola l'accuratezza media raggruppando per 'Test mode' e 'Model'
    mean_accuracy = df.groupby(['Test mode', 'Model'])['Accuracy (%)'].mean().unstack()

    # Definisci l'ordine desiderato per le modalità di test
    test_mode_order = ['zero', 'paraph', 'few', 'def', 'def-few']
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
    
def plot_dataset_size_comparison(csv_path, output_dir):
    """
    Genera un grafico a linee che confronta l'accuratezza dei modelli 
    al variare della dimensione del dataset di addestramento nella modalità zero shot.
    """
    # Carica i dati dal file CSV
    df = pd.read_csv(csv_path, sep=';').tail(18)  #regola tail in base agli ultimi test effettuati

    # Filtra per 'Test mode' == 'zero'
    df = df[df['Test mode'] == 'zero'].copy()

    # Estrai il nome del modello base e la dimensione del dataset
    def extract_info(model_name):
        parts = model_name.split('_trained_')
        base_model = parts[0]
        if len(parts) > 1 and parts[1].isdigit():
            return base_model, int(parts[1])
        return base_model, 0  # Modello base o non addestrato

    df[['base_model', 'dataset_size']] = df['Model'].apply(lambda x: pd.Series(extract_info(x)))

    # Calcola l'accuratezza media per ogni modello base e dimensione del dataset
    accuracy_by_size = df.groupby(['base_model', 'dataset_size'])['Accuracy (%)'].mean().reset_index()

    # Crea il grafico
    plt.figure(figsize=(14, 8))
    
    models = accuracy_by_size['base_model'].unique()
    colors = plt.get_cmap('tab10')

    for i, model in enumerate(models):
        model_data = accuracy_by_size[accuracy_by_size['base_model'] == model].sort_values('dataset_size')
        plt.plot(model_data['dataset_size'], model_data['Accuracy (%)'], marker='o', linestyle='-', label=model, color=colors(i))
        # Aggiungi etichette per ogni punto
        for x, y in zip(model_data['dataset_size'], model_data['Accuracy (%)']):
            plt.text(x, y + 0.5, f'{y:.2f}', ha='center', va='bottom', fontsize=8)

    # Aggiungi etichette e titolo
    plt.title('Accuratezza del Modello vs. Dimensione del Dataset di Addestramento (Test Mode: Zero-Shot)', fontsize=16)
    plt.xlabel('Dimensione del Dataset di Addestramento (numero di esempi)', fontsize=12)
    plt.ylabel('Accuratezza Media (%)', fontsize=12)
    plt.xticks(np.unique(accuracy_by_size['dataset_size'])) # Mostra tutte le dimensioni sull'asse x
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title='Modello Base')
    plt.ylim(bottom=min(80, accuracy_by_size['Accuracy (%)'].min() - 5), top=102)

    # Migliora il layout e salva il grafico
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'training_dataset_size_vs_accuracy.png')
    plt.savefig(output_path)

    print(f"Grafico delle dimensioni del dataset salvato in: {output_path}")
    plt.show()
    

def plot_epochs_vs_dataset_size(csv_path, output_dir, target_base_models):
    """
    Genera un grafico a linee che mostra il numero di epoche ottimale 
    al variare della dimensione del dataset di addestramento per i modelli specificati.
    """
    # Carica i dati dal file CSV
    df = pd.read_csv(csv_path, sep=';').tail(18) #regola tail in base agli ultimi test effettuati

    # Rimuovi righe dove 'Num Epochs' è nullo e converti in intero
    df = df.dropna(subset=['Num Epochs'])
    df['Num Epochs'] = df['Num Epochs'].astype(int)

    # Estrai il nome del modello base e la dimensione del dataset
    def extract_info(model_name):
        parts = model_name.split('_trained_')
        base_model = parts[0]
        if len(parts) > 1 and parts[1].isdigit():
            return base_model, int(parts[1])
        return None, None  # Ignora i modelli non addestrati o con formato errato

    df[['base_model', 'dataset_size']] = df['Model'].apply(lambda x: pd.Series(extract_info(x)))
    
    # Filtra per i modelli base target e per i modelli che sono stati addestrati
    df_filtered = df[df['base_model'].isin(target_base_models) & (df['dataset_size'].notna())].copy()

    if df_filtered.empty:
        print(f"Nessun dato trovato per i modelli {target_base_models} con informazioni sulle epoche.")
        return

    # Calcola la media delle epoche per ogni modello e dimensione del dataset
    epochs_by_size = df_filtered.groupby(['base_model', 'dataset_size'])['Num Epochs'].mean().reset_index()

    # Crea il grafico
    plt.figure(figsize=(14, 8))
    colors = plt.get_cmap('tab10')
    
    all_dataset_sizes = sorted(epochs_by_size['dataset_size'].unique())

    for i, model in enumerate(target_base_models):
        model_data = epochs_by_size[epochs_by_size['base_model'] == model].sort_values('dataset_size')
        if not model_data.empty:
            plt.plot(model_data['dataset_size'], model_data['Num Epochs'], marker='o', linestyle='-', label=model, color=colors(i))
            # Aggiungi etichette per ogni punto
            for x, y in zip(model_data['dataset_size'], model_data['Num Epochs']):
                plt.text(x, y + 0.1, f'{y:.1f}', ha='center', va='bottom', fontsize=9)

    # Aggiungi etichette e titolo
    plt.title('Numero di Epoche Ottimali vs. Dimensione Dataset', fontsize=16)
    plt.xlabel('Dimensione del Dataset di Addestramento (numero di esempi)', fontsize=12)
    plt.ylabel('Numero Medio di Epoche Ottimali', fontsize=12)
    plt.xticks(all_dataset_sizes)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title='Modello Base')

    # Migliora il layout e salva il grafico
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'epochs_vs_dataset_size.png')
    plt.savefig(output_path)

    print(f"Grafico delle epoche vs dimensione dataset salvato in: {output_path}")
    plt.show()


def plot_accuracy_loss_vs_epochs(csv_path, output_dir, target_base_models, invert_loss_axis=True, separate_graphs=False):
    """
    Genera grafici per accuratezza e loss vs. epoche.
    Può creare un grafico combinato a due assi o due grafici separati.
    """
    # Carica i dati dal file CSV
    df = pd.read_csv(csv_path, sep=';')
    
    # Rinomina colonne per coerenza
    if 'Accuracy (%)' in df.columns:
        df = df.rename(columns={'Accuracy (%)': 'train_accuracy'})

    # Estrai il nome del modello base
    def extract_base_model(model_name):
        return model_name.split('_trained_')[0]
    df['base_model'] = df['Model'].apply(extract_base_model)

    # Filtra per i modelli base di interesse
    df = df[df['base_model'].isin(target_base_models)].copy()

    if df.empty:
        print(f"Nessun dato trovato per i modelli {target_base_models} nel file {csv_path}.")
        return

    # Rimuovi righe dove i valori necessari sono nulli
    required_cols = ['Num Epochs', 'train_accuracy', 'eval_accuracy', 'Train Loss', 'Eval Loss']
    df = df.dropna(subset=required_cols)
    df['Num Epochs'] = df['Num Epochs'].astype(int)

    colors = plt.get_cmap('tab10')
    models = df['base_model'].unique()

    if separate_graphs:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12)) # Removed sharex=True, reduced height
        fig.suptitle('Loss and Accuracy (Train and Validation) vs. Number of Epochs', fontsize=14)

        # Grafico Loss (in alto)
        ax1.set_title('Train Loss vs. Validation Loss', fontsize=12)
        for i, model in enumerate(models):
            model_data = df[df['base_model'] == model].sort_values('Num Epochs')
            color = colors(i)
            ax1.plot(model_data['Num Epochs'], model_data['Train Loss'], color=color, linestyle='-', marker='o', label=f'{model} Train Loss')
            ax1.plot(model_data['Num Epochs'], model_data['Eval Loss'], color=color, linestyle='--', marker='x', label=f'{model} Validation Loss')
        
        ax1.set_ylabel('Loss', fontsize=10)
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax1.legend(loc='best')
        ax1.set_xlabel('Number of Epochs', fontsize=10)
        
        if invert_loss_axis:
            ax1.invert_yaxis()
            ax1.set_ylabel('Loss (Inverted)', fontsize=10)

        # Grafico Accuratezza (in basso)
        ax2.set_title('Train Accuracy vs. Validation Accuracy', fontsize=12)
        for i, model in enumerate(models):
            model_data = df[df['base_model'] == model].sort_values('Num Epochs')
            color = colors(i)
            ax2.plot(model_data['Num Epochs'], model_data['train_accuracy'], color=color, linestyle='-', marker='o', label=f'{model} Train Accuracy')
            ax2.plot(model_data['Num Epochs'], model_data['eval_accuracy'], color=color, linestyle='--', marker='x', label=f'{model} Validation Accuracy')

        ax2.set_xlabel('Number of Epochs', fontsize=10)
        ax2.set_ylabel('Accuracy (%)', fontsize=10)
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax2.legend(loc='best')
        ax2.set_ylim(bottom=min(80, df['train_accuracy'].min() - 5, df['eval_accuracy'].min() - 5), top=102)
        
        # Set x-ticks for both axes
        all_epochs = np.unique(df['Num Epochs'])
        ax1.set_xticks(all_epochs)
        ax2.set_xticks(all_epochs)

        output_filename = 'accuracy_loss_vs_epochs_separate.png'

    else: # Grafico combinato
        fig, ax1 = plt.subplots(figsize=(14, 8))
        ax2 = ax1.twinx()

        for i, model in enumerate(models):
            model_data = df[df['base_model'] == model].sort_values('Num Epochs').copy()
            color = colors(i)
            
            # Plot Accuracy su ax1
            ax1.plot(model_data['Num Epochs'], model_data['eval_accuracy'], color=color, linestyle='-', marker='o', label=f'{model} Validation Accuracy')
            
            # Plot Loss su ax2
            ax2.plot(model_data['Num Epochs'], model_data['Eval Loss'], color=color, linestyle='--', marker='x', label=f'{model} Validation Loss')

        # Imposta etichette e titoli
        ax1.set_xlabel('Number of Epochs', fontsize=12)
        ax1.set_ylabel('Validation Accuracy (%)', color='blue', fontsize=12)
        ax2.set_ylabel('Validation Loss', color='red', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='red')
        plt.title('Validation Accuracy vs. Validation Loss per Number of Epochs', fontsize=16)
        
        if invert_loss_axis:
            ax2.invert_yaxis()
            ax2.set_ylabel('Validation Loss (Inverted)', color='red', fontsize=12)

        # Unisci le legende dei due assi
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        output_filename = 'accuracy_loss_vs_epochs_combined.png'
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    if not separate_graphs:
        plt.xticks(np.unique(df['Num Epochs']))

    # Migliora il layout e salva il grafico
    plt.tight_layout(rect=[0, 0, 1, 0.97]) # Aggiusta per il suptitle
    if separate_graphs:
        fig.subplots_adjust(hspace=0.4)

    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path)

    print(f"Grafico di accuratezza e loss vs epoche salvato in: {output_path}")
    plt.show()

    
if __name__ == "__main__":
    mode_comparison_csv_path = 'accuracy_example_pool_sizes.csv'
    epoch_experiment_csv_path = 'epoch_vs_accuracy_loss.csv'
    output_dir = 'graphs/matplot'
    target_model_for_epochs_plot = ['gemma3_1b'] # Scegli i modelli base
    trained_model_size_for_comparison = 60 # Scegli la dimensione del training set da mostrare (es. 60) o None per tutti

    # Crea la cartella di output se non esiste
    os.makedirs(output_dir, exist_ok=True)
    # mode_comparison_graph(mode_comparison_csv_path, output_dir, trained_model_size_for_comparison)   #genera grafico di confronto tra modalità
    # plot_dataset_size_comparison(mode_comparison_csv_path, output_dir)
    # plot_epochs_vs_dataset_size(mode_comparison_csv_path, output_dir, target_model_for_epochs_plot)
    plot_accuracy_loss_vs_epochs(epoch_experiment_csv_path, output_dir, target_model_for_epochs_plot, invert_loss_axis=False, separate_graphs=True)

