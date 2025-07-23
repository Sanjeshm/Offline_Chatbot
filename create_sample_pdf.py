# Create sample PDF documents for testing the RAG system
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def create_sample_pdf():
    """Create a sample PDF document for testing"""
    filename = "/app/test_documents/ai_research_paper.pdf"
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=18,
        spaceAfter=30,
    )
    
    content = []
    
    # Title
    content.append(Paragraph("Advances in Large Language Models for Question Answering", title_style))
    content.append(Spacer(1, 20))
    
    # Abstract
    content.append(Paragraph("<b>Abstract</b>", styles['Heading2']))
    abstract_text = """This research paper explores the recent advances in large language models (LLMs) 
    specifically designed for question answering tasks. We investigate the effectiveness of transformer-based 
    architectures in understanding context and generating accurate responses. Our methodology involved training 
    models on diverse datasets and evaluating performance across multiple benchmarks. The key findings indicate 
    that models with attention mechanisms show superior performance in contextual understanding. We recommend 
    implementing retrieval-augmented generation (RAG) systems for enhanced accuracy in domain-specific applications."""
    content.append(Paragraph(abstract_text, styles['Normal']))
    content.append(Spacer(1, 20))
    
    # Introduction
    content.append(Paragraph("<b>1. Introduction</b>", styles['Heading2']))
    intro_text = """Large Language Models have revolutionized natural language processing tasks, particularly 
    in question answering systems. The main objective of this study is to analyze the performance characteristics 
    of different model architectures when applied to information retrieval tasks. Question answering systems have 
    evolved from simple keyword matching to sophisticated neural networks capable of understanding context and nuance."""
    content.append(Paragraph(intro_text, styles['Normal']))
    content.append(Spacer(1, 20))
    
    # Methodology
    content.append(Paragraph("<b>2. Methodology</b>", styles['Heading2']))
    method_text = """Our research methodology consisted of three main phases: data collection, model training, 
    and evaluation. We collected datasets from academic papers, technical documentation, and web sources. 
    The training process utilized distributed computing across multiple GPUs with batch sizes optimized for 
    memory efficiency. Performance evaluation metrics included accuracy, response time, and contextual relevance scores."""
    content.append(Paragraph(method_text, styles['Normal']))
    content.append(Spacer(1, 20))
    
    # Results
    content.append(Paragraph("<b>3. Results and Discussion</b>", styles['Heading2']))
    results_text = """The experimental results demonstrate significant improvements in question answering accuracy 
    when using retrieval-augmented approaches. Models achieved an average accuracy of 87% on benchmark datasets, 
    with response times under 2 seconds for most queries. The most effective approach combined semantic search 
    with generative models to provide both relevant and coherent answers."""
    content.append(Paragraph(results_text, styles['Normal']))
    content.append(Spacer(1, 20))
    
    # Conclusions
    content.append(Paragraph("<b>4. Conclusions</b>", styles['Heading2']))
    conclusion_text = """This study concludes that transformer-based language models, when combined with 
    retrieval mechanisms, provide superior performance for question answering tasks. The key recommendations 
    include using semantic chunking for document processing, implementing vector similarity search, and 
    optimizing model inference for real-time applications. Future work should focus on reducing computational 
    requirements while maintaining accuracy."""
    content.append(Paragraph(conclusion_text, styles['Normal']))
    
    # Build the PDF
    doc.build(content)
    return filename

if __name__ == "__main__":
    # Try to create PDF, if reportlab not available, create a text version
    try:
        filename = create_sample_pdf()
        print(f"Created sample PDF: {filename}")
    except ImportError:
        # Create a simple text file instead
        filename = "/app/test_documents/ai_research_paper.txt"
        with open(filename, 'w') as f:
            f.write("""Advances in Large Language Models for Question Answering

Abstract
This research paper explores the recent advances in large language models (LLMs) specifically designed for question answering tasks. We investigate the effectiveness of transformer-based architectures in understanding context and generating accurate responses. Our methodology involved training models on diverse datasets and evaluating performance across multiple benchmarks. The key findings indicate that models with attention mechanisms show superior performance in contextual understanding. We recommend implementing retrieval-augmented generation (RAG) systems for enhanced accuracy in domain-specific applications.

1. Introduction
Large Language Models have revolutionized natural language processing tasks, particularly in question answering systems. The main objective of this study is to analyze the performance characteristics of different model architectures when applied to information retrieval tasks. Question answering systems have evolved from simple keyword matching to sophisticated neural networks capable of understanding context and nuance.

2. Methodology
Our research methodology consisted of three main phases: data collection, model training, and evaluation. We collected datasets from academic papers, technical documentation, and web sources. The training process utilized distributed computing across multiple GPUs with batch sizes optimized for memory efficiency. Performance evaluation metrics included accuracy, response time, and contextual relevance scores.

3. Results and Discussion
The experimental results demonstrate significant improvements in question answering accuracy when using retrieval-augmented approaches. Models achieved an average accuracy of 87% on benchmark datasets, with response times under 2 seconds for most queries. The most effective approach combined semantic search with generative models to provide both relevant and coherent answers.

4. Conclusions
This study concludes that transformer-based language models, when combined with retrieval mechanisms, provide superior performance for question answering tasks. The key recommendations include using semantic chunking for document processing, implementing vector similarity search, and optimizing model inference for real-time applications. Future work should focus on reducing computational requirements while maintaining accuracy.
""")
        print(f"Created sample text file: {filename}")