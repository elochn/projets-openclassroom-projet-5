from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine("postgresql://localhost/churn_db")

Session = sessionmaker(bind=engine)
db = Session()
