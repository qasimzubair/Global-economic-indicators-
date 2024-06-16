from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QComboBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
import numpy as np
from statistics import mean, median, stdev
from sklearn.linear_model import LinearRegression
from scipy.stats import norm, poisson, kstest
import scipy.stats as stats
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QScrollArea, QWidget
import numpy as np
df = pd.read_csv('Global Economy Indicators.csv')


class MyGUI(QDialog):
def __init__(self):
super(MyGUI, self).__init__()
loadUi("untitled.ui", self)
self.pushButton_7.clicked.connect(self.show_graph)
self.pushButton_3.clicked.connect(self.show_descriptive)
self.pushButton_4.clicked.connect(self.show_probability)
self.pushButton_5.clicked.connect(self.show_regression)
self.pushButton_6.clicked.connect(self.show_confidence)


def show_graph(self):
self.hide()
graph_dialog = GraphDialog()
graph_dialog.exec_()

def show_descriptive(self):
self.hide()
descriptive_dialog = DescriptiveDialog()
descriptive_dialog.exec_()

def show_probability(self):
self.hide()
probability_dialog = ProbabilityDialog()
probability_dialog.exec_()

def show_regression(self):
self.hide()
regression_dialog = RegressionDialog()
regression_dialog.exec_()

def show_confidence(self):
self.hide()

confidence_dialog = ConfidenceDialog()
confidence_dialog.exec_()




class GraphDialog(QDialog):
def __init__(self):
super(GraphDialog, self).__init__()

loadUi("graph.ui", self)
self.country1 = ""
self.country2 = ""
self.f_country()
self.selected_countries_data = [self.country1, self.country2]
self.enter.clicked.connect(self.my)
self.home.clicked.connect(self.Home)
self.e.clicked.connect(self.comboBoxg2_changed)
self.comboBoxg.addItems(["AMA", "GDP", "Population_GDP", "GNI_Exports", 'Import_Export'])





def Home(self):
self.hide()
mygui = MyGUI()
mygui.exec_()
def my(self):

print("here again")
self.comboBoxg2.clear()

self.country1 = self.comboBoxg3.currentText()
self.country2 = self.comboBoxg4.currentText()


self.comboBoxg_changed()


def comboBoxg_changed(self):
self.selected_item = self.comboBoxg.currentText()

if self.selected_item == 'AMA':
self.comboBoxg2.addItems(["Options", "Histogram", "Line Graph"])

elif self.selected_item == 'GDP':
self.comboBoxg2.addItems(["Options", "Line Graph", "Histogram"])

elif self.selected_item == 'Population_GDP' or self.selected_item == 'GNI_Exports':
self.comboBoxg2.addItems(["Options", "Scatter Plot"])

elif self.selected_item == 'Import_Export':
self.comboBoxg2.addItems(["Options", "Box Plot"])




def comboBoxg2_changed(self):
var = self.comboBoxg.currentText()
graphName = self.comboBoxg2.currentText()

if graphName == "Histogram":
self.plot_histogram()
elif graphName == "Line Graph":
self.plot_Line_graph()
elif graphName =="Scatter Plot":
self.scatter_plot()
elif graphName =="Box Plot":
self.box_plot()

def scatter_plot(self):
if self.country1 != "Countries":
start_year = 1970
end_year = 2020
items = self.selected_item.split('_')
a, b = items

plt.figure(figsize=(12, 8))

data_filtered = df[(df['Country'] == self.country1) & (df['Year'] >= start_year) & (df['Year'] <= end_year)]

plot1 = sns.scatterplot(x=a, y=b, hue='Year', palette='viridis', data=data_filtered, s=100)

sns.regplot(x=a, y=b, data=data_filtered, scatter=False, color='gray')

correlation_coefficient = np.corrcoef(data_filtered[a], data_filtered[b])[0, 1]
correlation_text = f'Correlation: {correlation_coefficient:.2f}'
plt.text(0.95, 0.95, correlation_text, transform=plt.gca().transAxes, ha='right', va='top', fontsize=12)

norm = plt.Normalize(data_filtered['Year'].min(), data_filtered['Year'].max())
sm = cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])

cax = plt.gca().inset_axes([0.05, 0.1, 0.03, 0.8])
cbar = plt.colorbar(sm, cax=cax, label='Year',
orientation='vertical')

plt.title(f'Scatter Plot of {a} vs {b} for {self.country1} ({start_year}-{end_year})', fontsize=16)
plt.xlabel(a, fontsize=14)
plt.ylabel(b, fontsize=14)

handles1, labels1 = plot1.get_legend_handles_labels()

plt.legend(handles=handles1, labels=labels1, title='Year', loc='upper right', bbox_to_anchor=(1.25, 1),
fontsize=12)

plt.grid(True, linestyle='--', alpha=0.7)
sns.despine()

canvas = FigureCanvas(plt.gcf())

for i in reversed(range(self.verticalLayoutg.count())):
self.verticalLayoutg.itemAt(i).widget().setParent(None)

self.verticalLayoutg.addWidget(canvas)

def plot_Line_graph(self):

if self.country1 != "Countries" and self.country2 != "Countries":
plt.figure(figsize=(12, 8))

plot1 = sns.lineplot(x='Year', y=self.selected_item, data=df[df['Country'] == self.country1], label=self.country1,
linewidth=2, markers=True)

plot2 = sns.lineplot(x='Year', y=self.selected_item, data=df[df['Country'] == self.country2], label=self.country2,
linewidth=2, markers=True)

plt.title(f'{self.selected_item} Over Time for {self.country1} and {self.country2}', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel(f'{self.selected_item}', fontsize=14)

handles1, labels1 = plot1.get_legend_handles_labels()
handles2, labels2 = plot2.get_legend_handles_labels()

plt.legend(handles=handles1 + handles2, labels=labels1 + labels2, title='Country', loc='upper left',
fontsize=12)

plt.grid(True, linestyle='--', alpha=0.7)
sns.despine()

canvas = FigureCanvas(plt.gcf())

for i in reversed(range(self.verticalLayoutg.count())):
self.verticalLayoutg.itemAt(i).widget().setParent(None)

self.verticalLayoutg.addWidget(canvas)


def box_plot(self):
if self.country1 != "Countries":

plt.figure(figsize=(12, 8))

data_country = df[df['Country'] == self.country1][['Year', 'Imports', 'Exports']]

data_country_melted = data_country.melt(id_vars=['Year'], var_name='Transaction Type', value_name='Value')

data_country_melted['Decade'] = (data_country_melted['Year'] // 10) * 10

plot1 = sns.boxplot(x='Decade', y='Value', hue='Transaction Type', data=data_country_melted)

plt.title(f'Box Plot of Imports and Exports for {self.country1} (Grouped by Decades)', fontsize=16)
plt.xlabel('Decade', fontsize=14)
plt.ylabel('Value', fontsize=14)

plt.grid(True, linestyle='--', alpha=0.7)
sns.despine()

canvas = FigureCanvas(plt.gcf())

for i in reversed(range(self.verticalLayoutg.count())):
self.verticalLayoutg.itemAt(i).widget().setParent(None)

self.verticalLayoutg.addWidget(canvas)


def plot_histogram(self):
if self.country1 != "Countries":

plt.figure(figsize=(12, 8))

data_country = df[df['Country'] == self.country1][['Year', self.selected_item]]

data_country['Decade'] = (data_country['Year'] // 10) * 10
grouped_data = data_country.groupby('Decade')[self.selected_item].mean()

plt.bar(grouped_data.index, grouped_data.values, width=8, align='edge', edgecolor='black')

plt.title(f'Histogram of {self.selected_item} for {self.country1} (Year-wise with 10-Year Intervals)',
fontsize=16)
plt.xlabel('Decade', fontsize=14)
plt.ylabel(self.selected_item, fontsize=14)

plt.grid(True, linestyle='-', alpha=0.7)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

canvas = FigureCanvas(plt.gcf())

for i in reversed(range(self.verticalLayoutg.count())):
self.verticalLayoutg.itemAt(i).widget().setParent(None)

self.verticalLayoutg.addWidget(canvas)

class DescriptiveDialog(QDialog):
def __init__(self):
super(DescriptiveDialog, self).__init__()
loadUi("descriptive.ui", self)
self.country1 = ""
self.country2 = ""
self.h.clicked.connect(self.my)
self.home.clicked.connect(self.Home)
self.f_country()
self.comboBoxd4.addItems(['GDP', 'AMA', 'GNI', 'Imports', 'Exports'])

def Home(self):
self.hide()
mygui = MyGUI()
mygui.exec_()


def my(self):
self.country1 = self.comboBoxd3.currentText()
selected_item = self.comboBoxd4.currentText()

if self.country1 != 'Countries' :
if selected_item == 'GDP' or selected_item =='ANI' or selected_item =='GNI' or selected_item =='Imports' or selected_item =='Exports':
country_data = df[df['Country'] == self.country1][selected_item]

mean_value = np.mean(country_data)
median_value = np.median(country_data)
std_dev_value = np.std(country_data)
min_value = np.min(country_data)
max_value = np.max(country_data)
range_value = np.ptp(country_data)
iqr = np.percentile(country_data, 75) - np.percentile(country_data, 25)
skewness = country_data.skew()
kurtosis = country_data.kurtosis()

label_text = (
f'Mean: {mean_value}\n'
f'Median: {median_value}\n'
f'Standard Deviation: {std_dev_value}\n'
f'Minimum: {min_value}\n'
f'Maximum: {max_value}\n'
f'Range: {range_value}\n'
f'Interquartile Range: {iqr}\n'
)

self.label.setText(label_text)
self.label.show()



class ProbabilityDialog(QDialog):
def __init__(self):
super(ProbabilityDialog, self).__init__()
loadUi("Prob.ui", self)
self.f_country()
self.country1 = ""
self.comboBoxp2.addItems(['Options', 'IMF-Normal', 'Exports-Uniform'])
self.e.clicked.connect(self.my)

def Home(self):
self.hide()
mygui = MyGUI()
mygui.exec_()

def my(self):

self.country1 = self.comboBoxp.currentText()
selected_item = self.comboBoxp2.currentText()

low = float(self.textEdit.toPlainText())
high = float(self.textEdit_2.toPlainText())

print(low)
if self.country1 != 'Countries':
country_data = df[df['Country'] == self.country1]

if selected_item == 'Exports-Uniform':

self.analyze_uniform_distribution('Exports', country_data, 0, 5000000000)

elif selected_item == 'Changes_in_inventories':
self.analyze_poisson_distribution('Changes_in_inventories',country_data)

elif selected_item == 'IMF-Normal':
self.analyze_normal_distribution('IMF_Rate', country_data, low, high)



def analyze_normal_distribution(self, variable, country_data, range_low=None, range_high=None):
data = country_data[variable]
if range_low is None:
range_low = min(data)
if range_high is None:
range_high = max(data)

filtered_data = [value for value in data if range_low <= value <= range_high]

mean, std = np.mean(filtered_data), np.std(filtered_data)
normal_values = np.linspace(range_low, range_high, 100)
normal_cumulative_probabilities = stats.norm.cdf(normal_values, mean, std)

self.label.setText(f'Variable: {variable}\n'
f'Expected Vaalue: {mean}\n'
f'Standard Deviation: {std}\n'
f'Filtered Data: {filtered_data}\n'
f'Normal Distribution Probabilities: {normal_cumulative_probabilities}')


def analyze_uniform_distribution(self, variable, country_data, range_low=None, range_high=None):
data = country_data[variable]

if range_low is None:
range_low = min(data)
if range_high is None:
range_high = max(data)

filtered_data = [value for value in data if range_low <= value <= range_high]

mean = (range_low + range_high) / 2
range_diff = range_high - range_low

result_text = (f'Variable: {variable}\n'
f'Mean: {mean}\n'
f'Range: {range_low} to {range_high}\n'
f'Filtered Data: {filtered_data}\n'
f'Uniform Distribution Probability Density: 1 / {range_diff}')

self.label.setText(result_text)

def analyze_poisson_distribution(self, variable, country_data, range_low=None, range_high=None):
data = country_data[variable]

if range_low is None:
range_low = min(data)
if range_high is None:
range_high = max(data)

poisson_lambda = np.mean(data)
poisson_values = np.arange(range_low, range_high + 1)
poisson_probabilities = stats.poisson.pmf(poisson_values, poisson_lambda)

plt.hist(data, density=True, alpha=0.6, color='g', label='Histogram')
plt.plot(poisson_values, poisson_probabilities, 'r--', label='Poisson Distribution')
plt.title(f'Poisson Distribution Analysis for {variable}')
plt.xlabel(variable)
plt.ylabel('Probability Density')
plt.legend()
plt.show()


def f_country(self):

self.comboBoxp.addItems(['Countries', ' Afghanistan ', ' Albania ', ' Algeria ', ' Andorra ', ' Angola ',
' Antigua and Barbuda ', ' Azerbaijan ', ' Argentina ',
' Australia ', ' Austria ', ' Bahamas ', ' Bahrain ',
' Bangladesh ', ' Armenia ', ' Barbados ', ' Belgium ',
' Bermuda ', ' Bhutan ', ' Bolivia (Plurinational State of) ',
' Bosnia and Herzegovina ', ' Botswana ', ' Brazil ', ' Belize ',
' Solomon Islands ', ' British Virgin Islands ',
' Brunei Darussalam ', ' Bulgaria ', ' Myanmar ', ' Burundi ',
' Belarus ', ' Cambodia ', ' Cameroon ', ' Canada ',
' Cabo Verde ', ' Cayman Islands ', ' Central African Republic ',
' Sri Lanka ', ' Chad ', ' Chile ', ' China ', ' Colombia ',
' Comoros ', ' Congo ', ' D.R. of the Congo ', ' Cook Islands ',
' Costa Rica ', ' Croatia ', ' Cuba ', ' Cyprus ',
' Czechoslovakia (Former) ', ' Czechia ', ' Benin ', ' Denmark ',
' Dominica ', ' Dominican Republic ', ' Ecuador ', ' El Salvador ',
' Equatorial Guinea ', ' Ethiopia (Former) ', ' Ethiopia ',
' Eritrea ', ' Estonia ', ' Fiji ', ' Finland ', ' France ',
' French Polynesia ', ' Djibouti ', ' Gabon ', ' Georgia ',
' Gambia ', ' State of Palestine ', ' Germany ', ' Ghana ',
' Kiribati ', ' Greece ', ' Greenland ', ' Grenada ',
' Guatemala ', ' Guinea ', ' Guyana ', ' Haiti ', ' Honduras ',
' China, Hong Kong SAR ', ' Hungary ', ' Iceland ', ' India ',
' Indonesia ', ' Iran (Islamic Republic of) ', ' Iraq ',
' Ireland ', ' Israel ', ' Italy ', " Côte d'Ivoire ", ' Jamaica ',
' Japan ', ' Kazakhstan ', ' Jordan ', ' Kenya ',
' D.P.R. of Korea ', ' Republic of Korea ', ' Kosovo ', ' Kuwait ',
' Kyrgyzstan ', " Lao People's DR ", ' Lebanon ', ' Lesotho ',
' Latvia ', ' Liberia ', ' Libya ', ' Liechtenstein ',
' Lithuania ', ' Luxembourg ', ' China, Macao SAR ',
' Madagascar ', ' Malawi ', ' Malaysia ', ' Maldives ', ' Mali ',
' Malta ', ' Mauritania ', ' Mauritius ', ' Mexico ', ' Monaco ',
' Mongolia ', ' Republic of Moldova ', ' Montenegro ',
' Montserrat ', ' Morocco ', ' Mozambique ', ' Oman ', ' Namibia ',
' Nauru ', ' Nepal ', ' Netherlands ',
' Former Netherlands Antilles ', ' Aruba ', ' New Caledonia ',
' Vanuatu ', ' New Zealand ', ' Nicaragua ', ' Niger ',
' Nigeria ', ' Norway ', ' Micronesia (FS of) ',
' Marshall Islands ', ' Palau ', ' Pakistan ', ' Panama ',
' Papua New Guinea ', ' Paraguay ', ' Peru ', ' Philippines ',
' Poland ', ' Portugal ', ' Guinea-Bissau ', ' Timor-Leste ',
' Puerto Rico ', ' Qatar ', ' Romania ', ' Russian Federation ',
' Rwanda ', ' Saint Kitts and Nevis ', ' Anguilla ',
' Saint Lucia ', ' St. Vincent and the Grenadines ',
' San Marino ', ' Sao Tome and Principe ', ' Saudi Arabia ',
' Senegal ', ' Serbia ', ' Seychelles ', ' Sierra Leone ',
' Singapore ', ' Slovakia ', ' Viet Nam ', ' Slovenia ',
' Somalia ', ' South Africa ', ' Zimbabwe ',
' Yemen Democratic (Former) ', ' Spain ', ' Sudan (Former) ',
' Suriname ', ' Eswatini ', ' Sweden ', ' Switzerland ',
' Syrian Arab Republic ', ' Tajikistan ', ' Thailand ', ' Togo ',
' Tonga ', ' Trinidad and Tobago ', ' United Arab Emirates ',
' Tunisia ', ' Türkiye ', ' Turkmenistan ',
' Turks and Caicos Islands ', ' Tuvalu ', ' Uganda ', ' Ukraine ',
' North Macedonia ', ' USSR (Former) ', ' Egypt ',
' United Kingdom ', ' U.R. of Tanzania: Mainland ', ' Zanzibar ',
' United States ', ' Burkina Faso ', ' Uruguay ', ' Uzbekistan ',
' Venezuela (Bolivarian Republic of) ', ' Samoa ',
' Yemen Arab Republic (Former) ', ' Yemen ',
' Yugoslavia (Former) ', ' Zambia ', ' Curaçao ',
' Sint Maarten (Dutch part) ', ' South Sudan ', ' Sudan '])

class RegressionDialog(QDialog):
def __init__(self):
super(RegressionDialog, self).__init__()
loadUi("regression.ui", self)

self.f_country()

self.country1 = ""

self.comboBoxr.addItems(['Options', 'Population Prediction', 'GDP-Population', 'GNI-Exports'])

self.b.clicked.connect(self.my)


def my(self):
selected_item = self.comboBoxr.currentText()
self.country1 = self.comboBoxr2.currentText()
if self.country1 !='Countries':
if selected_item == 'Population Prediction':
country_data = df[df['Country'] == self.country1]

years = country_data['Year'].values.reshape(-1, 1)
population = country_data['Population'].values.reshape(-1, 1)

model = LinearRegression()

model.fit(years, population)

future_years = np.arange(1970, 2051, 20).reshape(-1, 1)
predicted_population_2050 = model.predict(future_years)

plt.figure(figsize=(12, 8))
plt.scatter(years, population, label='Actual Data', color='blue')
plt.plot(years, model.predict(years), label='Regression Line', color='red')
plt.scatter(future_years, predicted_population_2050, color='green', marker='x',
label='Predicted Population in 2050')
plt.xlabel('Year')
plt.ylabel('Population')
plt.title('Linear Regression: Population Prediction for 2050 with 20-Year Intervals')
plt.legend()

canvas = FigureCanvas(plt.gcf())

for i in reversed(range(self.verticalLayoutr.count())):
self.verticalLayoutr.itemAt(i).widget().setParent(None)

self.verticalLayoutr.addWidget(canvas)

canvas.draw()

elif selected_item =='GDP-Population':
country_data = df[df['Country'] == self.country1]

if country_data.empty:
raise ValueError(f"No data found for {self.country1}")

gdp = country_data['GDP'].values.reshape(-1, 1)
population = country_data['Population'].values.reshape(-1, 1)

country_data = df[df['Country'] ==self.country1]

years = country_data['Year'].values.reshape(-1, 1)
population = country_data['Population'].values.reshape(-1, 1)

model1 = LinearRegression()
model1.fit(years, population)
future_years = np.arange(2020, 2051, 10).reshape(-1, 1)
predicted_population = model1.predict(future_years)

population_2020 = predicted_population[0][0]
population_2030 = predicted_population[1][0]
population_2040 = predicted_population[2][0]
population_2050 = predicted_population[3][0]
model = LinearRegression()
model.fit(population, gdp)

new_population_values = np.array([population_2020, population_2030, population_2040, population_2050])

new_population_values = new_population_values.reshape(-1, 1)

predicted_gdp = model.predict(new_population_values)

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

sns.scatterplot(x=population.flatten(), y=gdp.flatten(), color='blue', label='Actual Data')

sns.lineplot(x=population.flatten(), y=model.predict(population).flatten(), color='red',
label='Linear Regression')

sns.scatterplot(x=new_population_values.flatten(), y=predicted_gdp.flatten(), color='green',
label='Predictions')

plt.xlabel('Population')
plt.ylabel('GDP')
plt.title(f'Linear Regression: GDP vs Population for {self.country1}')
plt.legend()

canvas = FigureCanvas(plt.gcf())

for i in reversed(range(self.verticalLayoutr.count())):
self.verticalLayoutr.itemAt(i).widget().setParent(None)

self.verticalLayoutr.addWidget(canvas)

canvas.draw()

elif selected_item == 'GNI-Exports':

country_data = df[df['Country'] == self.country1]

if country_data.empty:
raise ValueError(f"No data found for {self.country1}")

gdp = country_data['GNI'].values.reshape(-1, 1)
population = country_data['Exports'].values.reshape(-1, 1)

country_data = df[df['Country'] == self.country1]

years = country_data['Year'].values.reshape(-1, 1)
population = country_data['Exports'].values.reshape(-1, 1)

model1 = LinearRegression()
model1.fit(years, population)
future_years = np.arange(2020, 2051, 10).reshape(-1, 1)
predicted_population = model1.predict(future_years)

population_2020 = predicted_population[0][0]
population_2030 = predicted_population[1][0]
population_2040 = predicted_population[2][0]
population_2050 = predicted_population[3][0]
model = LinearRegression()
model.fit(population, gdp)

new_population_values = np.array([population_2020, population_2030, population_2040, population_2050])
new_population_values = new_population_values.reshape(-1, 1)
predicted_gdp = model.predict(new_population_values)
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

sns.scatterplot(x=population.flatten(), y=gdp.flatten(), color='blue', label='Actual Data')

sns.lineplot(x=population.flatten(), y=model.predict(population).flatten(), color='red',
label='Linear Regression')

sns.scatterplot(x=new_population_values.flatten(), y=predicted_gdp.flatten(), color='green',
label='Predictions')

plt.xlabel('Exports')
plt.ylabel('GNI')
plt.title(f'Linear Regression: GNI vs Exports for {self.country1}')
plt.legend()

canvas = FigureCanvas(plt.gcf())

for i in reversed(range(self.verticalLayoutr.count())):
self.verticalLayoutr.itemAt(i).widget().setParent(None)

self.verticalLayoutr.addWidget(canvas)

canvas.draw()







class ConfidenceDialog(QDialog):
def __init__(self):
super(ConfidenceDialog, self).__init__()
loadUi("Confidence.ui", self)
self.f_country()

self.comboBoxc.addItems(['Options', 'Population Prediction Confidence Interval',
'GDP-Population Confidence Interval', 'GNI-Exports Confidence Interval'])
self.b.clicked.connect(self.calculate_confidence_interval)

def calculate_confidence_interval(self):

print("A")
selected_item = self.comboBoxc.currentText()
country = self.c.currentText()

if selected_item == 'Population Prediction Confidence Interval':
country_data = df[df['Country'] == country]

years = country_data['Year'].values.reshape(-1, 1)
population = country_data['Population'].values.reshape(-1, 1)

model = LinearRegression()
model.fit(years, population)

slope, intercept = model.coef_[0][0], model.intercept_[0]
y_pred = model.predict(years)

residuals = population.flatten() - y_pred.flatten()
mse = np.sum(residuals ** 2) / len(residuals)
std_err_slope = np.sqrt(mse / np.sum((years - np.mean(years)) ** 2))
t_value = 1.96
margin_error = t_value * std_err_slope

lower_bound = slope - margin_error
upper_bound = slope + margin_error

self.label.setText(f"Confidence Interval for Population Prediction Slope: ({lower_bound}, {upper_bound})")
self.label.show()
elif selected_item == 'GDP-Population Confidence Interval':
country_data = df[df['Country'] == country]

years = country_data['Year'].values.reshape(-1, 1)
population = country_data['Population'].values.reshape(-1, 1)
gdp = country_data['GDP'].values.reshape(-1, 1)

model_population = LinearRegression()
model_population.fit(years, population)

model_gdp = LinearRegression()
model_gdp.fit(population, gdp)

def calculate_interval(model, x, y):
y_pred = model.predict(x)

residuals = y.flatten() - y_pred.flatten()
mse = np.sum(residuals ** 2) / len(residuals)
std_err_slope = np.sqrt(mse / np.sum((x - np.mean(x)) ** 2))
t_value = 1.96
margin_error = t_value * std_err_slope

slope, intercept = model.coef_[0][0], model.intercept_[0]
lower_bound = slope - margin_error
upper_bound = slope + margin_error

return lower_bound, upper_bound
lower_bound_gdp, upper_bound_gdp = calculate_interval(model_gdp, population, gdp)

self.label.setText(f"Confidence Interval for GDP-Population Slope: ({lower_bound_gdp}, {upper_bound_gdp})")
self.label.show()

elif selected_item == 'GNI-Exports Confidence Interval':
country_data = df[df['Country'] == country]

years = country_data['Year'].values.reshape(-1, 1)
gni = country_data['GNI'].values.reshape(-1, 1)
exports = country_data['Exports'].values.reshape(-1, 1)

model_gni = LinearRegression()
model_gni.fit(years, gni)

model_exports = LinearRegression()
model_exports.fit(exports, gni)

def calculate_interval(model, x, y):
y_pred = model.predict(x)

residuals = y.flatten() - y_pred.flatten()
mse = np.sum(residuals ** 2) / len(residuals)
std_err_slope = np.sqrt(mse / np.sum((x - np.mean(x)) ** 2))
t_value = 1.96
margin_error = t_value * std_err_slope

slope, intercept = model.coef_[0][0], model.intercept_[0]
lower_bound = slope - margin_error
upper_bound = slope + margin_error

return lower_bound, upper_bound


lower_bound_exports, upper_bound_exports = calculate_interval(model_exports, exports, gni)
self.label.setText(f"Confidence Interval for Exports-GNI Slope: ({lower_bound_exports}, {upper_bound_exports})")
self.label.show()

def main():
app = QApplication([])
window = MyGUI()

window.show()
app.exec_()

"""
def fit_distribution(data, column_name):
if "Agriculture" in column_name:
distribution = poisson
else:
distribution = norm

# Fit the distribution to the data
parameters = distribution.fit(data)

# Print the distribution name
print(f"Distribution applied to {column_name}: {distribution.name}")

"""

if __name__ == '__main__':
main()


"""
country_name = ' Pakistan '

fit_distribution(df['Population'], 'Population')
fit_distribution(df['Gross_National_Income(GNI)_in_USD'], 'GNI')
fit_distribution(df['GDP'], 'GDP')
fit_distribution(df['Agriculture_hunting_forestry_fishing_(ISIC_A-B)'], 'Agriculture')
fit_distribution(df['Imports'], 'Imports')
fit_distribution(df['Exports'], 'Exports')
fit_distribution(df['Manufacturing_(ISIC_D)'], 'Manufacturing')
"""
