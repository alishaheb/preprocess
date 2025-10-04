# hello_asyncio.py
import asyncio

async def say_after(delay, msg):
    await asyncio.sleep(delay)        # <-- non-blocking "pause"
    print(msg)

async def main():
    # run both coroutines concurrently
    t1 = asyncio.create_task(say_after(1, "one"))
    t2 = asyncio.create_task(say_after(2, "two"))
    await t1
    await t2

if __name__ == "__main__":
    asyncio.run(main())
