#!/usr/bin/env python3
"""
Generate figures for the restaking protocol security systematic review paper.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (10, 6)

# Figure 1: Protocol Timeline and TVL Growth
def generate_protocol_timeline():
    """Generate timeline of protocol launches and TVL growth"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Timeline data (months from Jan 2023)
    months = ['Jan 2023', 'Jul 2023', 'Jan 2024', 'Mar 2024', 'Jun 2024', 'Dec 2024']
    x_pos = [0, 6, 12, 14, 17, 23]
    
    # EigenLayer TVL (in billions)
    eigenlayer_tvl = [0, 0.5, 1.1, 10.4, 20.0, 20.0]
    
    # Other protocols (simplified)
    symbiotic_tvl = [0, 0, 0, 0, 0.3, 0.5]
    karak_tvl = [0, 0, 0, 0, 0.2, 0.3]
    babylon_tvl = [0, 0, 0, 0, 0.1, 0.1]
    
    ax.plot(x_pos, eigenlayer_tvl, 'o-', linewidth=2, markersize=8, label='EigenLayer', color='#6366f1')
    ax.plot(x_pos, symbiotic_tvl, 's-', linewidth=2, markersize=8, label='Symbiotic', color='#10b981')
    ax.plot(x_pos, karak_tvl, '^-', linewidth=2, markersize=8, label='Karak', color='#f59e0b')
    ax.plot(x_pos, babylon_tvl, 'd-', linewidth=2, markersize=8, label='Babylon', color='#ef4444')
    
    # Add launch markers
    ax.axvline(x=6, color='gray', linestyle='--', alpha=0.5, label='EigenLayer Launch')
    ax.text(6, 18, 'EigenLayer\nLaunch', ha='center', fontsize=8, alpha=0.7)
    
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('TVL (Billions USD)', fontsize=12, fontweight='bold')
    ax.set_title('Restaking Protocol Timeline and TVL Growth (2023-2024)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(months, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('figures/protocol_timeline.pdf', dpi=300, bbox_inches='tight')
    print("Generated: figures/protocol_timeline.pdf")
    plt.close()

# Figure 2: Security Landscape
def generate_security_landscape():
    """Generate security landscape visualization"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Threat categories
    threats = ['Dual-Layer\nSlashing', 'Smart Contract\nVulnerabilities', 
               'Centralization\nRisks', 'Liquidity\nChallenges', 
               'Cryptoeconomic\nAttacks']
    
    # Protocols
    protocols = ['EigenLayer', 'Symbiotic', 'Karak', 'Babylon']
    
    # Severity scores (1-5 scale, higher = more severe)
    # Format: [EigenLayer, Symbiotic, Karak, Babylon] for each threat
    severity_data = {
        'Dual-Layer\nSlashing': [5, 4, 4, 3],
        'Smart Contract\nVulnerabilities': [4, 5, 5, 4],
        'Centralization\nRisks': [4, 3, 3, 2],
        'Liquidity\nChallenges': [4, 4, 5, 2],
        'Cryptoeconomic\nAttacks': [3, 3, 4, 3]
    }
    
    # Create heatmap
    data = np.array([severity_data[threat] for threat in threats])
    
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=1, vmax=5)
    
    # Set ticks
    ax.set_xticks(np.arange(len(protocols)))
    ax.set_yticks(np.arange(len(threats)))
    ax.set_xticklabels(protocols)
    ax.set_yticklabels(threats)
    
    # Add text annotations
    for i in range(len(threats)):
        for j in range(len(protocols)):
            text = ax.text(j, i, data[i, j], ha="center", va="center", 
                          color="black", fontweight='bold', fontsize=11)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Threat Severity (1-5)', rotation=270, labelpad=20, fontsize=11)
    
    ax.set_title('Security Landscape: Threat Severity Across Restaking Protocols', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('figures/security_landscape.pdf', dpi=300, bbox_inches='tight')
    print("Generated: figures/security_landscape.pdf")
    plt.close()

# Figure 3: PRISMA Flow Diagram
def generate_prisma():
    """Generate PRISMA flow diagram"""
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.axis('off')
    
    # Box dimensions
    box_width = 3
    box_height = 0.8
    y_start = 10
    
    # Initial search
    rect1 = Rectangle((3.5, y_start), box_width, box_height, 
                     linewidth=2, edgecolor='black', facecolor='lightblue')
    ax.add_patch(rect1)
    ax.text(5, y_start + 0.4, 'Initial Search\n847 sources', 
           ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrow down
    ax.arrow(5, y_start, 0, -0.5, head_width=0.2, head_length=0.1, 
            fc='black', ec='black')
    
    # Title/Abstract screening
    y_screen = y_start - 1
    rect2 = Rectangle((3.5, y_screen), box_width, box_height, 
                     linewidth=2, edgecolor='black', facecolor='lightgreen')
    ax.add_patch(rect2)
    ax.text(5, y_screen + 0.4, 'Title/Abstract Screening\n235 remaining', 
           ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Excluded
    rect_ex1 = Rectangle((7.5, y_screen), 1.5, box_height, 
                        linewidth=1, edgecolor='black', facecolor='lightcoral')
    ax.add_patch(rect_ex1)
    ax.text(8.25, y_screen + 0.4, '612 excluded', 
           ha='center', va='center', fontsize=9)
    
    # Arrow down
    ax.arrow(5, y_screen, 0, -0.5, head_width=0.2, head_length=0.1, 
            fc='black', ec='black')
    
    # Full-text screening
    y_full = y_screen - 1
    rect3 = Rectangle((3.5, y_full), box_width, box_height, 
                     linewidth=2, edgecolor='black', facecolor='lightyellow')
    ax.add_patch(rect3)
    ax.text(5, y_full + 0.4, 'Full-Text Screening\n57 remaining', 
           ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Excluded
    rect_ex2 = Rectangle((7.5, y_full), 1.5, box_height, 
                        linewidth=1, edgecolor='black', facecolor='lightcoral')
    ax.add_patch(rect_ex2)
    ax.text(8.25, y_full + 0.4, '178 excluded', 
           ha='center', va='center', fontsize=9)
    
    # Arrow down
    ax.arrow(5, y_full, 0, -0.5, head_width=0.2, head_length=0.1, 
            fc='black', ec='black')
    
    # Final included
    y_final = y_full - 1
    rect4 = Rectangle((3.5, y_final), box_width, box_height, 
                     linewidth=2, edgecolor='black', facecolor='lightgreen')
    ax.add_patch(rect4)
    ax.text(5, y_final + 0.4, 'Final Included\n57 sources', 
           ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Breakdown
    y_break = y_final - 1.5
    breakdown_text = ('Academic: 12\nAudits: 8\nTechnical: 18\nDocs: 19')
    ax.text(5, y_break, breakdown_text, ha='center', va='top', 
           fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Title
    ax.text(5, y_start + 1.5, 'PRISMA Flow Diagram: Study Selection Process', 
           ha='center', va='center', fontsize=14, fontweight='bold')
    
    ax.set_xlim(0, 10)
    ax.set_ylim(y_break - 1, y_start + 2)
    
    plt.tight_layout()
    plt.savefig('figures/prisma.pdf', dpi=300, bbox_inches='tight')
    print("Generated: figures/prisma.pdf")
    plt.close()

if __name__ == '__main__':
    import os
    os.makedirs('figures', exist_ok=True)
    
    print("Generating figures for restaking protocol security review...")
    generate_protocol_timeline()
    generate_security_landscape()
    generate_prisma()
    print("\nAll figures generated successfully!")
