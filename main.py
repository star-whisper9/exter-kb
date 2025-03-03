from fastapi import FastAPI
from api import router

import config

app = FastAPI()
app.include_router(router.router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.APP_HOST, port=config.APP_PORT)
