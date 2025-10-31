"""
Memory visualization module.

Provides visual representations of memory networks and search paths.
"""
import time
from typing import List, Dict, Any, Optional, Tuple
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich import box


class MemoryVisualizer:
    """
    Visualizes memory networks and search paths.
    """

    def __init__(self):
        """Initialize the visualizer."""
        self.console = Console()

    def visualize_memory_graph(
        self,
        units: List[Dict[str, Any]],
        links: List[Dict[str, Any]],
        output_file: str = "memory_graph.png",
        highlight_nodes: Optional[List[str]] = None,
    ):
        """
        Create a visual representation of the memory graph.

        Args:
            units: List of memory units (id, text, context, etc.)
            links: List of links (from_unit_id, to_unit_id, link_type, weight)
            output_file: Output file path for the visualization
            highlight_nodes: Optional list of node IDs to highlight
        """
        # Create directed graph
        G = nx.DiGraph()

        # Add nodes
        node_labels = {}
        for unit in units:
            unit_id = str(unit['id'])
            # Truncate text for display
            label = unit['text'][:40] + "..." if len(unit['text']) > 40 else unit['text']
            G.add_node(unit_id)
            node_labels[unit_id] = label

        # Add edges
        temporal_edges = []
        semantic_edges = []
        for link in links:
            from_id = str(link['from_unit_id'])
            to_id = str(link['to_unit_id'])
            weight = link['weight']
            link_type = link['link_type']

            if link_type == 'temporal':
                temporal_edges.append((from_id, to_id, weight))
            else:  # semantic
                semantic_edges.append((from_id, to_id, weight))

            G.add_edge(from_id, to_id, weight=weight, type=link_type)

        # Create figure
        fig, ax = plt.subplots(figsize=(20, 14))
        ax.set_facecolor('#1a1a2e')
        fig.patch.set_facecolor('#0f0f1e')

        # Use spring layout for better visualization
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        # Draw temporal edges (blue)
        if temporal_edges:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=[(e[0], e[1]) for e in temporal_edges],
                edge_color='#4ecdc4',
                alpha=0.6,
                width=2,
                arrows=True,
                arrowsize=15,
                arrowstyle='->',
                connectionstyle='arc3,rad=0.1',
                ax=ax
            )

        # Draw semantic edges (purple)
        if semantic_edges:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=[(e[0], e[1]) for e in semantic_edges],
                edge_color='#ff6b9d',
                alpha=0.6,
                width=2,
                arrows=True,
                arrowsize=15,
                arrowstyle='->',
                connectionstyle='arc3,rad=0.1',
                ax=ax
            )

        # Determine node colors
        node_colors = []
        for node in G.nodes():
            if highlight_nodes and node in highlight_nodes:
                node_colors.append('#ffd93d')  # Yellow for highlighted
            else:
                node_colors.append('#6c63ff')  # Purple for normal

        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=3000,
            alpha=0.9,
            ax=ax
        )

        # Draw labels
        nx.draw_networkx_labels(
            G, pos,
            node_labels,
            font_size=8,
            font_color='white',
            font_weight='bold',
            ax=ax
        )

        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color='#4ecdc4', lw=2, label='Temporal Links'),
            plt.Line2D([0], [0], color='#ff6b9d', lw=2, label='Semantic Links'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#6c63ff',
                      markersize=10, label='Memory Unit', linestyle=''),
        ]
        if highlight_nodes:
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ffd93d',
                          markersize=10, label='Highlighted', linestyle='')
            )

        ax.legend(handles=legend_elements, loc='upper left', facecolor='#2d2d44',
                 edgecolor='white', fontsize=10, labelcolor='white')

        # Title
        ax.set_title('Memory Network Graph\nTemporal + Semantic Architecture',
                    color='white', fontsize=16, fontweight='bold', pad=20)

        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, facecolor='#0f0f1e')
        plt.close()

        self.console.print(f"[green]✓[/green] Memory graph saved to [cyan]{output_file}[/cyan]")


