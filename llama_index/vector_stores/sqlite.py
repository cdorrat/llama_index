import struct
from collections import namedtuple
from typing import Any, List, Optional, Type

from llama_index.schema import MetadataMode, TextNode
from llama_index.vector_stores.types import (MetadataFilters,
                                             NodeWithEmbedding, VectorStore,
                                             VectorStoreQuery,
                                             VectorStoreQueryResult)
from llama_index.vector_stores.utils import (metadata_dict_to_node,
                                             node_to_metadata_dict)


class SqliteVectorStore(VectorStore):
   stores_text = True
   flat_metadata = False

   def __init__(
         self, connection_string: str, async_connection_string: str, table_name: str
   ) -> None:
      try:
         import sqlite3  # noqa: F401
         import aiosqlite  # noqa: F401
         import sqlite_vss  # noqa: F401
      except ImportError:
         raise ImportError(
               "`sqlite3`, `aiosqlite` and `sqlite_vss` packages should be preinstalled"
         )
      
      self.connection_string = connection_string
      self.async_connection_string = async_connection_string
      lower_name = table_name.lower()
      self.node_table_name: str = f'{lower_name}_data'
      self.vss_table_name: str = f'{lower_name}_vss'
      self._connect()
      self._add_extension()
      self._create_tables_if_not_exists()

   async def close(self) -> None:
      self.con.close()

   def _connect(self) -> Any:
      import sqlite3  # noqa: F401     

      self.con = sqlite3.connect(self.connection_string)

   def _add_extension(self) -> None:
      import sqlite_vss

      self.con.enable_load_extension(True)
      sqlite_vss.load(self.con)
      self.con.enable_load_extension(False)

   def _create_tables_if_not_exists(self) -> None:
      with self.con.cursor() as cx:
         cx.execute(
            f'''create table if not exists {self.node_table_name}(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_text TEXT not null,
            metadata JSON,
            node_id TEXT,
            doc_id TEXT
            )'''
         )
         cx.execute(f'create virtual table if not exists {self.vss_table_name} '
                  ' using vss0(embedding(1536))')

   def add(self, embedding_results: List[NodeWithEmbedding]) -> List[str]:
      ids = []
      with self.con.cursor() as cx:
         for result in embedding_results:
            metadata = node_to_metadata_dict(result.node, 
                                              remove_text=True, 
                                              flat_metadata=self.flat_metadata)
            cx.execute(f'insert into {self.node_table_name}(node_id, node_text, metadata, doc_id)'
                       ' values (?, ?, ?, ?)', 
                       (result.node.id, 
                        result.node.get_content(metadata_mode=MetadataMode.NONE), 
                        metadata,
                        metadata['doc_id']))
            cx.execute(f'insert into {self.node_table_name}(rowid, embedding)',
                       (cx.lastrowid, self._embedding_to_db(result.embedding)))
            ids.append(result.id)
      cx.close()
      self.con.commit()
      return ids
   
   def _embedding_to_db(ebbedding) -> Any:
      # json.dumps(result.embedding) # alternate json version
      return struct.pack('%sf' % len(ebbedding), *ebbedding)


   def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
      with self.con.cursor() as cx:
         cx.execute(f'delete from {self.vss_table_name} '
                     f' where rowid in (select id from {self.node_table_name} where doc_id = ?)',
                     (ref_doc_id))
         cx.execute(f'from {self.node_table_name} where doc_id = ?', (ref_doc_id))
      cx.close()
      self.con.commit()

   def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
      cx = self.con.cursor()
      results = cx.fetchall(f'''select n.node_id, n.node_text, n.metadata, n.doc_id, vss.distance
                  from {self.node_table_name} n join                  
                    (select rowid, distance 
                     from vss_lookup 
                     where vss_search(embedding, ?)
                     limit ?) vss on n.id = vss.rowid ''',
                     (self._embedding_to_db(query.query_embedding), query.similarity_top_k))
      return self._db_query_to_query_result(results)
      
      
   def _db_query_to_query_result(db_results) -> VectorStoreQueryResult:
      nodes = []
      similarities = []
      ids = []
      for res in db_results:
         node = metadata_dict_to_node(res['metadata'])
         node.set_content(str(res['node_text']))
         nodes.append()
         similarities.append[res['distance']]
         ids.append[res['node_id']]

      return VectorStoreQueryResult(
            nodes=nodes,
            similarities=similarities,
            ids=ids,
        )



