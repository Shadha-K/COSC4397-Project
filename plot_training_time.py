import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the training time data for all implementations
data = pd.read_csv('training_time.csv')
data_shmem = pd.read_csv('training_time_shmem.csv')
data_regtil = pd.read_csv('training_time_regtil.csv')  # Added register tiling implementation

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))  # Increased width to accommodate the textbox

# Plot 1: Cumulative GPU time comparison
ax1.plot(data['Epoch'], data['Time_on_GPU'], 'o-', color='blue', linewidth=2, markersize=8, label='Naive Implementation')
ax1.plot(data_shmem['Epoch'], data_shmem['Time_on_GPU'], 's-', color='red', linewidth=2, markersize=8, label='Shared Memory Implementation')
ax1.plot(data_regtil['Epoch'], data_regtil['Time_on_GPU'], '^-', color='green', linewidth=2, markersize=8, label='Register Tiling Implementation')  # Added register tiling plot
ax1.set_title('Cumulative GPU Processing Time Comparison', fontsize=16)
ax1.set_xlabel('Epoch', fontsize=14)
ax1.set_ylabel('Cumulative Time (seconds)', fontsize=14)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.set_xticks(data['Epoch'][::2])  # Plot every other epoch on x-axis to avoid crowding
ax1.legend(loc='upper left')

# Calculate per-epoch time (difference in cumulative times)
# Fix the SettingWithCopyWarning by correctly creating and modifying dataframes
data_per_epoch = data.copy()
data_shmem_per_epoch = data_shmem.copy()
data_regtil_per_epoch = data_regtil.copy()

# Calculate per-epoch time for all implementations
data_per_epoch['Per_Epoch_Time'] = data_per_epoch['Time_on_GPU'].diff()
data_per_epoch.loc[0, 'Per_Epoch_Time'] = data_per_epoch['Time_on_GPU'].iloc[0]  # First epoch time

data_shmem_per_epoch['Per_Epoch_Time'] = data_shmem_per_epoch['Time_on_GPU'].diff()
data_shmem_per_epoch.loc[0, 'Per_Epoch_Time'] = data_shmem_per_epoch['Time_on_GPU'].iloc[0]  # First epoch time

data_regtil_per_epoch['Per_Epoch_Time'] = data_regtil_per_epoch['Time_on_GPU'].diff()
data_regtil_per_epoch.loc[0, 'Per_Epoch_Time'] = data_regtil_per_epoch['Time_on_GPU'].iloc[0]  # First epoch time

# Annotate points for all implementations
ax1.annotate(f"{data['Time_on_GPU'].iloc[-1]:.2f}s", 
             (data['Epoch'].iloc[-1], data['Time_on_GPU'].iloc[-1]),
             textcoords="offset points", 
             xytext=(0,10), 
             ha='center',
             color='blue')
             
ax1.annotate(f"{data_shmem['Time_on_GPU'].iloc[-1]:.2f}s", 
             (data_shmem['Epoch'].iloc[-1], data_shmem['Time_on_GPU'].iloc[-1]),
             textcoords="offset points", 
             xytext=(0,10), 
             ha='center',
             color='red')

ax1.annotate(f"{data_regtil['Time_on_GPU'].iloc[-1]:.2f}s",  # Added register tiling annotation
             (data_regtil['Epoch'].iloc[-1], data_regtil['Time_on_GPU'].iloc[-1]),
             textcoords="offset points", 
             xytext=(0,10), 
             ha='center',
             color='green')

# Plot 2: Error rate comparison
ax2.plot(data['Epoch'], data['Error'], 'o-', color='blue', linewidth=2, markersize=8, label='Naive Implementation')
ax2.plot(data_shmem['Epoch'], data_shmem['Error'], 's-', color='red', linewidth=2, markersize=8, label='Shared Memory Implementation')
ax2.plot(data_regtil['Epoch'], data_regtil['Error'], '^-', color='green', linewidth=2, markersize=8, label='Register Tiling Implementation')  # Added register tiling plot
ax2.set_title('Error Rate Comparison', fontsize=16)
ax2.set_xlabel('Epoch', fontsize=14)
ax2.set_ylabel('Error', fontsize=14)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.set_xticks(data['Epoch'][::2])  # Plot every other epoch on x-axis to avoid crowding
ax2.legend(loc='upper right')

# Use log scale for error visualization if values span multiple orders of magnitude
if max(max(data['Error']), max(data_shmem['Error']), max(data_regtil['Error'])) / min(min(data['Error'] + 1e-10), min(data_shmem['Error'] + 1e-10), min(data_regtil['Error'] + 1e-10)) > 100:
    ax2.set_yscale('log')
    ax2.set_ylabel('Error (log scale)', fontsize=14)

# Annotate only the first and last points to avoid clutter
for df, color in [(data, 'blue'), (data_shmem, 'red'), (data_regtil, 'green')]:  # Added register tiling to annotation loop
    ax2.annotate(f"{df['Error'].iloc[0]:.2e}", 
                 (df['Epoch'].iloc[0], df['Error'].iloc[0]),
                 textcoords="offset points", 
                 xytext=(0,10), 
                 ha='center',
                 color=color)
    ax2.annotate(f"{df['Error'].iloc[-1]:.2e}", 
                 (df['Epoch'].iloc[-1], df['Error'].iloc[-1]),
                 textcoords="offset points", 
                 xytext=(0,10), 
                 ha='center',
                 color=color)

# Calculate speedups
if data['Time_on_GPU'].iloc[-1] > 0:
    speedup_shmem = data['Time_on_GPU'].iloc[-1] / data_shmem['Time_on_GPU'].iloc[-1]
    speedup_regtil = data['Time_on_GPU'].iloc[-1] / data_regtil['Time_on_GPU'].iloc[-1]  # Added register tiling speedup
    speedup_regtil_vs_shmem = data_shmem['Time_on_GPU'].iloc[-1] / data_regtil['Time_on_GPU'].iloc[-1]  # Added speedup comparison
else:
    speedup_shmem = 0
    speedup_regtil = 0
    speedup_regtil_vs_shmem = 0

# Add text box with comparison statistics to the right of the plots
# Creating text for the comparison textbox
comparison_text = '\n'.join([
    f"Naive Implementation:",
    f"  Total GPU Time: {data['Time_on_GPU'].iloc[-1]:.2f}s",
    f"  Final Error: {data['Error'].iloc[-1]:.2e}",
    f"  Error Reduction: {(1 - data['Error'].iloc[-1]/data['Error'].iloc[0])*100:.1f}%",
    f"",
    f"Shared Memory Implementation:",
    f"  Total GPU Time: {data_shmem['Time_on_GPU'].iloc[-1]:.2f}s",
    f"  Final Error: {data_shmem['Error'].iloc[-1]:.2e}",
    f"  Error Reduction: {(1 - data_shmem['Error'].iloc[-1]/data_shmem['Error'].iloc[0])*100:.1f}%",
    f"",
    f"Register Tiling Implementation:",  # Added register tiling stats
    f"  Total GPU Time: {data_regtil['Time_on_GPU'].iloc[-1]:.2f}s",
    f"  Final Error: {data_regtil['Error'].iloc[-1]:.2e}",
    f"  Error Reduction: {(1 - data_regtil['Error'].iloc[-1]/data_regtil['Error'].iloc[0])*100:.1f}%",
    f"",
    f"Performance Improvement:",
    f"  Shared Memory vs. Naive: {speedup_shmem:.2f}x",
    f"  Register Tiling vs. Naive: {speedup_regtil:.2f}x",
    f"  Register Tiling vs. Shared Memory: {speedup_regtil_vs_shmem:.2f}x"
])

# Create a dedicated axes for the text box
fig.subplots_adjust(right=0.7)  # Make room for the text box
text_ax = fig.add_axes([0.72, 0.35, 0.25, 0.3])  # [left, bottom, width, height]
text_ax.axis('off')  # Hide the axes

# Add the text box to the figure
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
text_ax.text(0, 0.5, comparison_text, fontsize=10, verticalalignment='center', bbox=props)

plt.tight_layout(rect=[0, 0, 0.7, 1])  # Adjust layout to leave space for the text box
plt.savefig('training_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("Training performance comparison visualization saved as 'training_performance_comparison.png'")