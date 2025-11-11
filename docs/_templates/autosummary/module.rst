{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}
   :members:
   :undoc-members: False
   :special-members: __call__, __init__
   :private-members: False
   :imported-members: False
   :show-inheritance:
   :member-order: bysource

   {% block modules %}
   {% if modules %}
   .. rubric:: Modules

   .. autosummary::
   :toctree:
   :template: autosummary/module.rst
   :recursive:

   {% for item in modules %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block functions %}
   {% if functions %}

   {% endif %}
   {% endblock %}

   {% block classes %}
   {% if classes %}

   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}
   .. rubric:: Exceptions

   .. autosummary::
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
