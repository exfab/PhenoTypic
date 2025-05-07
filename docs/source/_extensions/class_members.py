from docutils.parsers.rst import Directive, directives
from docutils import nodes
from sphinx.util.docutils import SphinxDirective
from sphinx.addnodes import pending_xref
import inspect
import importlib
from docutils.statemachine import StringList
import re

class ClassMembersDirective(SphinxDirective):
    """
    Directive to automatically document public methods, properties, and attributes of a class
    with docstring summaries in a two-column table format.
    """
    has_content = True
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {
        'methods': directives.flag,
        'properties': directives.flag,
        'attributes': directives.flag,
        'exclude': directives.unchanged,
    }

    def get_docstring_summary(self, obj):
        """Extract the first line/sentence from the docstring"""
        if not obj.__doc__:
            return ""
        
        # Get the first paragraph of the docstring
        docstring = obj.__doc__.strip().split('\n\n')[0]
        # Get the first sentence if possible
        first_sentence = re.split(r'\. ', docstring, 1)[0]
        return first_sentence.strip()

    def get_signature(self, method):
        """Get the method signature as a string"""
        try:
            sig = inspect.signature(method)
            # Get the return annotation if available
            return_annotation = sig.return_annotation
            if return_annotation is not inspect.Signature.empty:
                # Format the return annotation in a shorter form
                if hasattr(return_annotation, '__name__'):
                    return_str = f" -> {return_annotation.__name__}"
                else:
                    # Convert the full type annotation to a shorter form
                    return_str = f" -> {str(return_annotation)}"
                    # Replace common patterns with shorter versions
                    # return_str = return_str.replace('typing.', '')
                    # return_str = return_str.replace('matplotlib.axes._axes.Axes', 'Axes')
                    # return_str = return_str.replace('matplotlib.figure.Figure', 'Figure')
                    # return_str = return_str.replace('<class \'', '')
                    # return_str = return_str.replace('\'>', '')
                    # return_str = return_str.replace('numpy.', 'np.')
                    # return_str = return_str.replace('pandas.', 'pd.')
                    # return_str = return_str.replace('self, ', '')
                return "(...)" + return_str
            else:
                return "(...)"
        except (ValueError, TypeError):
            return "(...)"

    def run(self):
        class_path = self.arguments[0]
        module_name, class_name = class_path.rsplit('.', 1)
        
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            self.state.document.reporter.warning(
                f'Error importing {class_path}: {e}')
            return []

        exclude = self.options.get('exclude', '').split(',')
        exclude = [name.strip() for name in exclude if name.strip()]

        # Collect methods
        methods_info = []
        if 'methods' in self.options or not self.options:
            for name, member in inspect.getmembers(cls):
                # Skip private methods (but include special methods)
                if name.startswith('_') and not (name.startswith('__') and name.endswith('__')):
                    continue
                if name in exclude:
                    continue
                if inspect.isfunction(member) or inspect.ismethod(member):
                    methods_info.append({
                        'name': name,
                        'fullname': f"{class_name}.{name}",
                        'obj': member,
                        'signature': self.get_signature(member),
                        'summary': self.get_docstring_summary(member)
                    })

        # Collect properties
        properties_info = []
        if 'properties' in self.options or not self.options:
            for name, member in inspect.getmembers(cls):
                if name.startswith('_'):
                    continue
                if name in exclude:
                    continue
                if isinstance(member, property):
                    # Get the docstring from the property getter if available
                    summary = self.get_docstring_summary(member.fget if member.fget else member)
                    properties_info.append({
                        'name': name,
                        'fullname': f"{class_name}.{name}",
                        'obj': member,
                        'summary': summary
                    })

        # Collect other attributes
        attributes_info = []
        if 'attributes' in self.options or not self.options:
            for name, member in inspect.getmembers(cls):
                if name.startswith('_'):
                    continue
                if name in exclude:
                    continue
                if not (inspect.isfunction(member) or inspect.ismethod(member) or 
                        isinstance(member, property)):
                    attributes_info.append({
                        'name': name,
                        'fullname': f"{class_name}.{name}",
                        'obj': member,
                        'summary': str(member) if not hasattr(member, '__doc__') or not member.__doc__ 
                                  else self.get_docstring_summary(member)
                    })

        # Create a single container for all sections to ensure proper ordering
        container = nodes.container()
        sections = []
        
        # Create a section for attributes
        if attributes_info:
            section = nodes.section()
            section['ids'] = ['attributes']
            title = nodes.title('', 'Attributes')
            section += title
            
            # Create a table for attributes
            table = nodes.table()
            tgroup = nodes.tgroup(cols=2)
            table += tgroup
            
            # Add column specifications
            tgroup += nodes.colspec(colwidth=40)
            tgroup += nodes.colspec(colwidth=60)
            
            # Add table header
            thead = nodes.thead()
            tgroup += thead
            header_row = nodes.row()
            header_row += nodes.entry('', nodes.paragraph('', 'Attribute'))
            header_row += nodes.entry('', nodes.paragraph('', 'Description'))
            thead += header_row
            
            # Add table body
            tbody = nodes.tbody()
            tgroup += tbody
            
            for attr in attributes_info:
                row = nodes.row()
                
                # Attribute name
                name_entry = nodes.entry()
                name_para = nodes.paragraph()
                
                # Create a reference that links to the actual documentation
                name_ref = pending_xref('', refdomain='py', reftype='attr',
                                        reftarget=attr['fullname'], refexplicit=True)
                name_ref += nodes.Text(attr['name'])
                name_para += name_ref
                name_entry += name_para
                row += name_entry
                
                # Attribute description
                desc_entry = nodes.entry()
                desc_para = nodes.paragraph('', attr['summary'])
                desc_entry += desc_para
                row += desc_entry
                
                tbody += row
            
            section += table
            sections.append(('attributes', section))
        
        # Create a section for properties
        if properties_info:
            section = nodes.section()
            section['ids'] = ['properties']
            title = nodes.title('', 'Properties')
            section += title
            
            # Create a table for properties
            table = nodes.table()
            tgroup = nodes.tgroup(cols=2)
            table += tgroup
            
            # Add column specifications
            tgroup += nodes.colspec(colwidth=40)
            tgroup += nodes.colspec(colwidth=60)
            
            # Add table header
            thead = nodes.thead()
            tgroup += thead
            header_row = nodes.row()
            header_row += nodes.entry('', nodes.paragraph('', 'Property'))
            header_row += nodes.entry('', nodes.paragraph('', 'Description'))
            thead += header_row
            
            # Add table body
            tbody = nodes.tbody()
            tgroup += tbody
            
            for prop in properties_info:
                row = nodes.row()
                
                # Property name
                name_entry = nodes.entry()
                name_para = nodes.paragraph()
                
                # Create a reference that links to the actual documentation
                name_ref = pending_xref('', refdomain='py', reftype='attr',
                                        reftarget=prop['fullname'], refexplicit=True)
                name_ref += nodes.Text(prop['name'])
                name_para += name_ref
                name_entry += name_para
                row += name_entry
                
                # Property description
                desc_entry = nodes.entry()
                desc_para = nodes.paragraph('', prop['summary'])
                desc_entry += desc_para
                row += desc_entry
                
                tbody += row
            
            section += table
            sections.append(('properties', section))

        # Create a section for methods
        if methods_info:
            section = nodes.section()
            section['ids'] = ['methods']
            title = nodes.title('', 'Methods')
            section += title
            
            # Create a table for methods
            table = nodes.table()
            tgroup = nodes.tgroup(cols=2)
            table += tgroup
            
            # Add column specifications
            tgroup += nodes.colspec(colwidth=40)
            tgroup += nodes.colspec(colwidth=60)
            
            # Add table header
            thead = nodes.thead()
            tgroup += thead
            header_row = nodes.row()
            header_row += nodes.entry('', nodes.paragraph('', 'Method'))
            header_row += nodes.entry('', nodes.paragraph('', 'Description'))
            thead += header_row
            
            # Add table body
            tbody = nodes.tbody()
            tgroup += tbody
            
            for method in methods_info:
                row = nodes.row()
                
                # Method name and signature
                name_entry = nodes.entry()
                name_para = nodes.paragraph()
                
                # Create a reference that links to the actual documentation
                name_ref = pending_xref('', refdomain='py', reftype='meth',
                                        reftarget=method['fullname'], refexplicit=True)
                name_ref += nodes.Text(method['name'])
                name_para += name_ref
                name_para += nodes.Text(method['signature'])
                name_entry += name_para
                row += name_entry
                
                # Method description
                desc_entry = nodes.entry()
                desc_para = nodes.paragraph('', method['summary'])
                desc_entry += desc_para
                row += desc_entry
                
                tbody += row
            
            section += table
            sections.append(('methods', section))
        

        
        # Assemble sections in the desired order: attributes, properties, methods
        result = []
        
        # Define the order of sections
        section_order = ['attributes','properties', 'methods']
        
        # Create a dictionary of sections for easy lookup
        section_dict = dict(sections)
        
        # Add sections to result in the specified order
        for section_id in section_order:
            if section_id in section_dict:
                result.append(section_dict[section_id])
        
        return result

def setup(app):
    print("\n\n*** Loading enhanced class_members extension v0.8 ***\n\n")
    app.add_directive('class-members', ClassMembersDirective)
    
    return {
        'version': '0.8',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }