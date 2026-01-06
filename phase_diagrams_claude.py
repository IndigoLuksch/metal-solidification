#%matplotlib inline
import matplotlib.pyplot as plt
from pycalphad import Database, binplot
import pycalphad.variables as v
import anthropic
import os
from dotenv import load_dotenv
load_dotenv()

from config import PHASE_DIAGRAMS_DIR

filename = 'Fe-Si-Zn.tdb'
filepath = PHASE_DIAGRAMS_DIR + '/' + filename
components = ['FE', 'SI']

def load_phase_diagram(tdb_file, components):
    """Load TDB database once at initialization"""
    with open(tdb_file, encoding='iso-8859-1') as f:
        dbf = Database(f.read())
    return dbf

def get_liquidus_solidus(dbf, components, X_solute, pressure=101325):
    """Query liquidus and solidus temperatures for given composition"""
    # Use pycalphad equilibrium calculations
    # Returns T_liquidus, T_solidus for the composition
    pass

dbf = load_phase_diagram(filepath, components)
l_s = get_liquidus_solidus(dbf, components, 0.1)
print(l_s)




USER_PROMPT = f'''Create a binary phase diagram:
-Database: {filepath}
-Components: {components}
-Temperature range: 300K to appropriate upper limit
-Composition range: 0 to 1 wt%
-Pressure: 101325 Pa
-Try loading the databse with the following encodings: latin-1, iso-8859-1, cp1252
Generate complete runnable code following the pycalphad binplot structure.'''
'''

client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

message = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=20000,
    temperature=1,
    system="You are an expert in computational thermodynamics using pycalphad. Generate clean, well-documented code for binary phase diagram calculations. Always:\n-Load TDB files using: with open(filename, encoding='iso-8859-1') as f: dbf = Database(f.read())\n-Auto-detect phases with list(dbf.phases.keys())\n-Include 'VA' in components\n-Use binplot() for visualization\n-Label temperature in Kelvin",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": USER_PROMPT
                }
            ]
        }
    ]
)

response_text = message.content[0].text

# Extract code from markdown if present
import re
code_match = re.search(r'```python\n(.*?)\n```', response_text, re.DOTALL)
if code_match:
    generated_code = code_match.group(1)
else:
    generated_code = response_text

# Execute it
print(generated_code)
exec(generated_code)
'''