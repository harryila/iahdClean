"""
Clean visualization: Accuracy vs Context Length
"""
import json
import matplotlib.pyplot as plt

with open('llama3_results/head/context_attention_sweep.json', 'r') as f:
    data = json.load(f)

# Extract and sort data
results = []
for ctx_len, res in data['by_context_length'].items():
    results.append({
        'ctx': int(ctx_len),
        'unshuf': res['unshuffled']['accuracy'],
        'shuf': res['shuffled']['accuracy']
    })

results.sort(key=lambda x: x['ctx'])
ctx = [r['ctx'] for r in results]
unshuf = [r['unshuf'] * 100 for r in results]  # Convert to percentage
shuf = [r['shuf'] * 100 for r in results]  # Convert to percentage

# Print to verify
print("Context | Unshuffled | Shuffled")
for i, r in enumerate(results):
    print(f"{r['ctx']:>7} | {unshuf[i]:>10.1f}% | {shuf[i]:.1f}%")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(ctx, unshuf, 'o-', color='#2563eb', linewidth=2.5, markersize=8, label='Unshuffled')
plt.plot(ctx, shuf, 's--', color='#dc2626', linewidth=2.5, markersize=8, label='Shuffled')

plt.xlabel('Context Length (tokens)', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Llama 3 Retrieval Accuracy vs Context Length', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.ylim(-5, 75)
plt.xscale('log')
plt.xticks(ctx, [f'{x//1000}K' if x >= 1000 else str(x) for x in ctx], rotation=45)

plt.tight_layout()
plt.savefig('llama3_results/head/accuracy_vs_context.png', dpi=150, facecolor='white')
print("\nSaved: llama3_results/head/accuracy_vs_context.png")

