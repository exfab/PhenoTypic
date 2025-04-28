{%- if title is defined %}
.. rubric:: {{ title }}
{%- endif %}

{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoproperty:: {{ objname }}

{% set accessor_class = objname.split('.')[-1] %}
{% set accessor_module = "phenoscope.core.accessors" %}
{% set class_map = {
    "array": "ImageArray",
    "matrix": "ImageMatrix",
    "enh_matrix": "ImageEnhancedMatrix",
    "objmap": "ObjectMap",
    "objmask": "ObjectMask",
    "objects": "ObjectsAccessor",
    "hsv": "HsvAccessor",
    "grid": "GridAccessor"
} %}

{% if accessor_class in class_map %}
.. currentmodule:: {{ accessor_module }}

.. class-members:: {{ accessor_module }}.{{ class_map[accessor_class] }}
   :attributes:
   :properties:
   :methods:

.. rubric:: {{ class_map[accessor_class] }} API Reference
{# Generate the class documentation with autoclass, showing all members directly #}
.. autoclass:: {{ class_map[accessor_class] }}
   :members:
   :show-inheritance:
   :special-members: __getitem__, __setitem__, __len__, __call__
   :member-order: groupwise
   :noindex:
{% endif %}