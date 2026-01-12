"""
Agentic RAG System - CLI Entry Point
Provides command-line interface for document ingestion and querying.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from typing import Optional, List

# Initialize CLI app
app = typer.Typer(
    name="rag",
    help="Agentic RAG System with Hybrid Retrieval",
    add_completion=False
)

console = Console()


@app.command()
def ingest(
    path: str = typer.Argument(..., help="Path to document or directory to ingest"),
    clear: bool = typer.Option(False, "--clear", "-c", help="Clear existing data before ingesting")
):
    """
    Ingest documents into the RAG system.
    
    Processes documents, generates embeddings, and builds the knowledge graph.
    """
    from src.orchestrator import get_orchestrator
    from src.db.qdrant_client import QdrantManager
    from src.db.neo4j_client import Neo4jManager
    
    console.print(Panel.fit(
        "[bold blue]Document Ingestion[/bold blue]",
        subtitle="Agentic RAG System"
    ))
    
    # Validate path
    target_path = Path(path)
    if not target_path.exists():
        console.print(f"[red]Error: Path does not exist: {path}[/red]")
        raise typer.Exit(1)
    
    # Clear existing data if requested
    if clear:
        console.print("[yellow]Clearing existing data...[/yellow]")
        try:
            QdrantManager().clear_collection()
            Neo4jManager().clear_graph()
            console.print("[green]Data cleared successfully[/green]")
        except Exception as e:
            console.print(f"[red]Error clearing data: {e}[/red]")
    
    # Run ingestion
    console.print(f"\n[cyan]Processing: {path}[/cyan]")
    
    try:
        orchestrator = get_orchestrator()
        result = orchestrator.ingest_documents([str(target_path)])
        
        # Display results
        table = Table(title="Ingestion Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Documents Processed", str(len(result.raw_documents)))
        table.add_row("Chunks Created", str(len(result.chunks)))
        table.add_row("Entities Extracted", str(len(result.entities)))
        table.add_row("Relationships Found", str(len(result.relationships)))
        table.add_row("Indexed to Qdrant", "✓" if result.indexed_to_qdrant else "✗")
        table.add_row("Indexed to Neo4j", "✓" if result.indexed_to_neo4j else "✗")
        
        console.print(table)
        
        if result.errors:
            console.print("\n[yellow]Warnings/Errors:[/yellow]")
            for error in result.errors:
                console.print(f"  • {error}")
        
        console.print("\n[green]Ingestion complete![/green]")
        
    except Exception as e:
        console.print(f"[red]Error during ingestion: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask the RAG system")
):
    """
    Query the RAG system.
    
    Performs hybrid retrieval and generates a grounded answer with citations.
    """
    from src.orchestrator import get_orchestrator
    
    console.print(Panel.fit(
        "[bold blue]RAG Query[/bold blue]",
        subtitle="Agentic RAG System"
    ))
    
    console.print(f"\n[cyan]Query:[/cyan] {question}")
    console.print("\n[dim]Processing...[/dim]\n")
    
    try:
        orchestrator = get_orchestrator()
        result = orchestrator.query(question)
        
        # Display answer
        formatted_answer = orchestrator.format_answer(result)
        
        console.print(Panel(
            Markdown(formatted_answer),
            title="[bold green]Answer[/bold green]",
            border_style="green" if result.verified else "yellow"
        ))
        
        # Display retrieval stats
        stats_table = Table(title="Retrieval Statistics")
        stats_table.add_column("Source", style="cyan")
        stats_table.add_column("Results", style="green")
        
        stats_table.add_row("Qdrant (Semantic)", str(len(result.qdrant_results)))
        stats_table.add_row("Neo4j (Graph)", str(len(result.neo4j_results)))
        stats_table.add_row("Merged Evidence", str(len(result.merged_evidence)))
        
        console.print(stats_table)
        
        if result.errors:
            console.print("\n[yellow]Warnings:[/yellow]")
            for error in result.errors:
                console.print(f"  • {error}")

        # Debug: Show top evidence content
        console.print("\n[bold cyan]Top Evidence Snippets:[/bold cyan]")
        for i, chunk in enumerate(result.merged_evidence[:5]):
            is_validated = chunk.metadata.get("graph_validated", False)
            badge = "[bold green][GRAPH VALIDATED][/bold green] " if is_validated else "[dim][SEMANTIC ONLY][/dim] "
            
            text_preview = chunk.text[:200].replace("\n", " ") + "..."
            console.print(f"{i+1}. {badge} {text_preview}")
        
    except Exception as e:
        console.print(f"[red]Error during query: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(1)


@app.command()
def status():
    """
    Show the status of the RAG system.
    
    Displays information about indexed documents and knowledge graph.
    """
    from src.db.qdrant_client import QdrantManager
    from src.db.neo4j_client import Neo4jManager
    
    console.print(Panel.fit(
        "[bold blue]System Status[/bold blue]",
        subtitle="Agentic RAG System"
    ))
    
    # Qdrant status
    console.print("\n[cyan]Qdrant Vector Store:[/cyan]")
    try:
        qdrant_info = QdrantManager().get_collection_info()
        if "error" in qdrant_info:
            console.print(f"  [red]Error: {qdrant_info['error']}[/red]")
        else:
            console.print(f"  Collection: {qdrant_info.get('name', 'N/A')}")
            console.print(f"  Points: {qdrant_info.get('points_count', 0)}")
            console.print(f"  Status: {qdrant_info.get('status', 'unknown')}")
    except Exception as e:
        console.print(f"  [red]Connection error: {e}[/red]")
    
    # Neo4j status
    console.print("\n[cyan]Neo4j Knowledge Graph:[/cyan]")
    try:
        neo4j_info = Neo4jManager().get_graph_stats()
        if "error" in neo4j_info:
            console.print(f"  [red]Error: {neo4j_info['error']}[/red]")
        else:
            console.print(f"  Entities: {neo4j_info.get('total_entities', 0)}")
            console.print(f"  Relationships: {neo4j_info.get('total_relationships', 0)}")
            
            if neo4j_info.get('entities_by_type'):
                console.print("  Entity Types:")
                for etype, count in neo4j_info['entities_by_type'].items():
                    console.print(f"    - {etype}: {count}")
    except Exception as e:
        console.print(f"  [red]Connection error: {e}[/red]")


@app.command()
def clear(
    qdrant: bool = typer.Option(False, "--qdrant", "-q", help="Clear only Qdrant data"),
    neo4j: bool = typer.Option(False, "--neo4j", "-n", help="Clear only Neo4j data")
):
    """
    Clear data from the RAG system.
    
    By default (no flags), clears BOTH Qdrant and Neo4j.
    Use --qdrant or --neo4j to clear specific databases.
    """
    from src.db.qdrant_client import QdrantManager
    from src.db.neo4j_client import Neo4jManager
    
    # If no specific flags, clear both
    clear_all = not qdrant and not neo4j
    
    console.print(Panel.fit(
        "[bold red]Clear Data[/bold red]",
        subtitle="Agentic RAG System"
    ))
    
    if clear_all:
        msg = "Are you sure you want to clear ALL data (Qdrant + Neo4j)?"
    elif qdrant and neo4j:
        msg = "Are you sure you want to clear Qdrant AND Neo4j data?"
    elif qdrant:
        msg = "Are you sure you want to clear ONLY Qdrant data?"
    else:
        msg = "Are you sure you want to clear ONLY Neo4j data?"
    
    if not typer.confirm(msg):
        console.print("[yellow]Cancelled[/yellow]")
        raise typer.Exit()
    
    if clear_all or qdrant:
        console.print("\n[yellow]Clearing Qdrant...[/yellow]")
        try:
            QdrantManager().clear_collection()
            console.print("[green]Qdrant cleared[/green]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    if clear_all or neo4j:
        console.print("\n[yellow]Clearing Neo4j...[/yellow]")
        try:
            Neo4jManager().clear_graph()
            console.print("[green]Neo4j cleared[/green]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    console.print("\n[green]Done![/green]")


@app.command()
def interactive():
    """
    Start an interactive query session.
    
    Allows continuous querying without restarting.
    """
    from src.orchestrator import get_orchestrator
    
    console.print(Panel.fit(
        "[bold blue]Interactive Mode[/bold blue]\n"
        "Type your questions and press Enter.\n"
        "Type 'exit' or 'quit' to end the session.",
        subtitle="Agentic RAG System"
    ))
    
    try:
        orchestrator = get_orchestrator()
    except Exception as e:
        console.print(f"[red]Error initializing: {e}[/red]")
        raise typer.Exit(1)
    
    while True:
        console.print()
        try:
            question = typer.prompt("Question")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Goodbye![/yellow]")
            break
        
        if question.lower() in ['exit', 'quit', 'q']:
            console.print("[yellow]Goodbye![/yellow]")
            break
        
        if not question.strip():
            continue
        
        try:
            result = orchestrator.query(question)
            formatted_answer = orchestrator.format_answer(result)
            
            console.print(Panel(
                Markdown(formatted_answer),
                title="[bold green]Answer[/bold green]",
                border_style="green" if result.verified else "yellow"
            ))
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    app()
