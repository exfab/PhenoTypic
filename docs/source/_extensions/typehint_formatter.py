"""
Sphinx extension to format type hints consistently.
"""

def setup(app):
    """
    Setup function for the extension.
    """
    # Connect to the autodoc-process-signature event
    app.connect('autodoc-process-signature', process_signature)
    
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }

def process_signature(app, what, name, obj, options, signature, return_annotation):
    """
    Process the signature to clean up type hints.
    """
    # Define replacements for common problematic type hints
    replacements = {
        'matplotlib.axes._axes.Axes': 'matplotlib.axes.Axes',
        '<class \'matplotlib.axes._axes.Axes\'>': 'matplotlib.axes.Axes',
        'matplotlib.figure.Figure': 'matplotlib.figure.Figure',
        '<class \'matplotlib.figure.Figure\'>': 'matplotlib.figure.Figure',
    }
    
    # Process return annotation
    if return_annotation:
        for old, new in replacements.items():
            if old in return_annotation:
                return_annotation = return_annotation.replace(old, new)
    
    # Process signature
    if signature:
        for old, new in replacements.items():
            if old in signature:
                signature = signature.replace(old, new)
    
    return signature, return_annotation
