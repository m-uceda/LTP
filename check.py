import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from matplotlib.font_manager import FontProperties

rcParams['font.family'] = 'Segoe UI Emoji'

# Example plot
plt.plot([1, 2, 3], [4, 5, 6])
emoji_font = FontProperties(family="Segoe UI Emoji")
plt.title("Test Title with 😊 Emoji", fontproperties=emoji_font)
plt.show()