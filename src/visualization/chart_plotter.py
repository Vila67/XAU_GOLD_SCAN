import matplotlib.pyplot as plt
import seaborn as sns

class GoldChartPlotter:
    def __init__(self):
        # Utiliser un style matplotlib standard au lieu de seaborn
        plt.style.use('classic')
        # Configurer seaborn
        sns.set_theme()
        
    def plot_price_history(self, df):
        """Trace l'historique des prix avec volume"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[3, 1])
        
        # Prix
        ax1.plot(df['Date'], df['Close'], label='Prix de clôture')
        ax1.set_title('Historique du prix de l\'or')
        ax1.set_ylabel('Prix (USD)')
        ax1.grid(True)
        ax1.legend()
        
        # Volume
        ax2.bar(df['Date'], df['Volume'], color='gray', alpha=0.5)
        ax2.set_ylabel('Volume')
        ax2.grid(True)
        
        # Formater les dates
        fig.autofmt_xdate()
        
        plt.tight_layout()
        return fig 

    def plot_technical_indicators(self, df):
        """Trace les indicateurs techniques sur 3 graphiques"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))
        
        # Moyennes mobiles
        ax1.plot(df['Date'], df['Close'], label='Prix', alpha=0.5)
        ax1.plot(df['Date'], df['SMA_20'], label='SMA 20')
        ax1.plot(df['Date'], df['SMA_50'], label='SMA 50')
        ax1.plot(df['Date'], df['SMA_200'], label='SMA 200')
        ax1.set_title('Prix et Moyennes Mobiles')
        ax1.grid(True)
        ax1.legend()
        
        # RSI
        ax2.plot(df['Date'], df['RSI'], label='RSI')
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Survente')
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Surachat')
        ax2.set_title('RSI')
        ax2.grid(True)
        ax2.legend()
        
        # MACD
        ax3.plot(df['Date'], df['MACD'], label='MACD')
        ax3.plot(df['Date'], df['Signal_Line'], label='Signal')
        ax3.bar(df['Date'], df['MACD'] - df['Signal_Line'], alpha=0.3, 
                color=['red' if x < 0 else 'green' for x in df['MACD'] - df['Signal_Line']])
        ax3.set_title('MACD')
        ax3.grid(True)
        ax3.legend()
        
        # Formater les dates
        fig.autofmt_xdate()
        plt.tight_layout()
        return fig 