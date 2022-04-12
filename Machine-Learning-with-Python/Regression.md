# Regression

## Linear Regression

### Introduction to Regression

what is regression ?
Regression is process of predicting a continuous value.

| ENGINESIZE | CYLINDERS | FUELCONSUMPTION_COMB | CO2EMISSIONS |
|--- |---|--- |---|
| 0 | 2.0 | 4 | 8.5 | 196 |
| 1 | 2.4 | 4 | 9.6 | 221 |
| 2 | 1.5 | 4 | 5.9 | 136 |
| 3 | 3.5 | 6 |11.1 | 255 |
| 4 | 3.5 | 6 |10.6 | 244 |
| 5 | 3.5 | 6 |10.0 | 230 |
| 6 | 3.5 | 6 |10.1 | 232 |
| 7 | 3.7 | 6 |11.1 | 255 |
| 8 | 3.7 | 6 |11.6 | 267 |
| 9 | 2.4 | 4 | 9.2 | ? | <- Let's predict co2emission for new car

In regression there are two types of variable: a dependent variable and one or more independent variables.

Dependent variable: The dependent variable can be seen as the state, target or final goal we study and try to predict. It is notated by Y.
Independent variable: It is also called as explanatory variables, can be seen as the causes of these states. It is notated by X.
A regression model relates Y to a function of X.
In above dataset, ENGINESIZE, CYLINDERS, FUELCONSUMPTION_COMB are dependent variable and CO@EMISSIONS is independent variable. 
In regression columns are called as features. The key point in regression is that our dependent value should be continuous and can not be a discrete value.
However the independent variable can be measured on either a categorical or continuous measurement scale.

Using above data, we are going to build regression estimation model, this model is going to predict the expected co2 emission for a new or unknown car.

There are two type of regression models: simple regression model and multiple regression.

Simple Linear Regression: One independent variable is used to estimate a dependent variable. It can be either linear or non linear. eg: predicting co2 emission using the variable of engine size. Linearity of regression is based on the nature of relationship between dependent and independent variable.

Multiple Regression: When more than one independent variable is present the process is called multiple linear regression. eg: predicting co2 emission using engine size and the number of cylinders in any given car.

Depending on the relationship between dependent or independent variables it can be linear or non-linear.

Applications of regression:
- Sales forecasting
- Satisfaction analysis
- Price estimation
- Employement income

### Simple Linear Regression:
