import httpx2._decoders as d

d.SUPPORTED_DECODERS.pop("br", None)
print(d.SUPPORTED_DECODERS)

from music_rag import answer_query

answer=answer_query("Which songs mix well together?", songs_dir=r"C:\Users\karal\Music\PioneerDJ\Downloads")
print(answer)
