# Quadratic Gaussian Year-on-year inflation (QGY) model
## About the paper
This python code is a straightforward implementation of the following work:

Trovato, Manlio, Diana Ribeiro, and Hringur Gretarsson. Quadratic gaussian inflation. Risk 25.9 (2012): 86.

Gretarsson, Hringur, A Quadratic Gaussian Year-on-Year Inflation Model (January 31, 2013). PhD Thesis, Imperial College London.

For verification purpose, some of the results in the paper are replicated using the python code.

## Code structure
The code consists of several classes and test files.

QgyModel.py: the main class of QGY model, containing all the parameters.

QgySwapPricer.py: class for pricing inflation swaplet

QgyCapFloorPricer.py: class for pricing caplet and floorlet

QgyVolSurface.py: class for imply Year-on-year and zero-coupon volatility surface by inputing the forward caplet price

QgIntegration.py: class for computing Gaussian integration, which is called in the caplet pricer.

Test(\*).py: other files starting with Test is for testing the class or replicating the results in the paper.

