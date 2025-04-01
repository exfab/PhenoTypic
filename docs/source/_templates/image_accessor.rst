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
    "objmap": "ImageObjects",
    "objmask": "ImageObjects",
    "object": "ImageObjects",
    "hsv": "HsvAccessor",
    "grid": "GridAccessor"
} %}

{% if accessor_class in class_map %}
.. currentmodule:: {{ accessor_module }}

.. autoclass:: {{ class_map[accessor_class] }}
   :private-members: False 
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
{% endif %}
