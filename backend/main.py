from datetime import datetime, timedelta, timezone
from typing import Annotated
from fastapi.responses import FileResponse

from fastapi import Depends, FastAPI, HTTPException, status,File,UploadFile
#from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
#from jose import JWTError, jwt
#from passlib.context import CryptContext
from pydantic import BaseModel
import os
from random import randint
import uuid

#from pymongo import MongoClient

from fastapi.middleware.cors import CORSMiddleware

from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')
# Image DIR
IMAGEDIR = "images/"


class URL(BaseModel):
    url:str

app = FastAPI()

origins = [
    "http://localhost:3000",
]



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)



'''
#Database connection
client = MongoClient('mongodb://127.0.0.1:27017')
mydb = client['FakeLogo']
myusers = mydb['users']


# to get a string like this run:
# openssl rand -hex 32
SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 100

class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str | None = None


class User(BaseModel):
    firstname:str
    lastname:str
    username:str
    hashed_password:str
    disabled: bool | None = None


class UserInDB(User):
    hashed_password: str


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")




def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def get_user(myusers, username: str):
    user_dict = myusers.find_one({'username':username})

    if user_dict:
        return UserInDB(**user_dict)


def authenticate_user(myusers, username: str, password: str):
    user = get_user(myusers, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(myusers, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)]
):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def create_user(user: User):
    hashed_password = pwd_context.hash(user.hashed_password)
    user_dict = user.dict()
    user_dict["hashed_password"] = hashed_password
    user_dict["disabled"] = False
    result = myusers.insert_one(user_dict)
    return result.inserted_id


@app.post("/signup")
def register(user: User):
    # Check if the username or phone already exists
    existing_user = myusers.find_one({"username": user.username})
    if existing_user:
            raise HTTPException(status_code=400, detail="Username already exists")

    try:
        user_id = create_user(user)
        return {"message": "User created successfully", "user_id": str(user_id)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/token")
async def login_for_access_token(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]) -> Token:
    user = authenticate_user(myusers, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")


@app.get("/users/me/", response_model=User)
async def read_users_me(current_user: Annotated[User, Depends(get_current_active_user)]):
    return current_user


@app.get("/users/me/items/")
async def read_own_items(current_user: Annotated[User, Depends(get_current_active_user)]):
    return [{"item_id": "Foo", "owner": current_user.username}]
'''




@app.post('/predict')
def predict(url:URL):

    image_url = url.url
    # Run batched inference on a list of images
    try:
        results = model(image_url)  # return a list of Results object
    except:
        raise HTTPException(status_code=400, detail="Invalid image url..."
                                                    "The image should be jpg,png")

    if not results:
        raise HTTPException(status_code=400, detail="No results found")


    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        #result.show()  # display to screen
        result.save(filename='result.jpg')

    return FileResponse('result.jpg')



@app.post("/upload")
async def create_upload_file(file: UploadFile = File(...)):
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()

    # save the file
    with open(f"{IMAGEDIR}{file.filename}", "wb") as f:
        f.write(contents)



    path = f.name

    try:
        results = model(path)  # return a list of Results object
    except:
        return {'message':'Invalid image url'}

    if not results:
        raise HTTPException(status_code=400, detail="No results found")


    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        #result.show()  # display to screen
        result.save(filename='result.jpg')

    return FileResponse('result.jpg')







