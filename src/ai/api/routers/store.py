"""Cross-thread long-term memory Store endpoints scoped strictly by Flux `space_id`."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from langgraph.store.base import BaseStore, Item, SearchItem

from ai.api.dependencies import get_memory_store, verify_internal_key
from ai.api.schemas import (
    SpaceStoreDeleteResponse,
    StoreDeleteResponse,
    StoreItemResponse,
    StorePutRequest,
    StoreSearchRequest,
    StoreSearchResponse,
)

router = APIRouter(tags=["store"], dependencies=[Depends(verify_internal_key)])


def _space_namespace(space_id: str, namespace: str) -> tuple[str, str, str]:
    return ("spaces", space_id, namespace)


def _to_item_response(item: Item | SearchItem, *, space_id: str, namespace: str) -> StoreItemResponse:
    raw_score = getattr(item, "score", None)
    return StoreItemResponse(
        space_id=space_id,
        namespace=namespace,
        key=item.key,
        value=dict(item.value),
        created_at=item.created_at.isoformat() if item.created_at else None,
        updated_at=item.updated_at.isoformat() if item.updated_at else None,
        score=float(raw_score) if isinstance(raw_score, int | float) else None,
    )


@router.put("/store/items", response_model=StoreItemResponse)
async def put_store_item(
    payload: StorePutRequest,
    store: Annotated[BaseStore, Depends(get_memory_store)],
) -> StoreItemResponse:
    """Create or update a persistent long-term memory item inside a space namespace."""
    ns = _space_namespace(payload.space_id, payload.namespace)
    await store.aput(ns, payload.key, payload.value)
    stored = await store.aget(ns, payload.key)
    if stored is None:
        return StoreItemResponse(
            space_id=payload.space_id,
            namespace=payload.namespace,
            key=payload.key,
            value=payload.value,
        )
    return _to_item_response(stored, space_id=payload.space_id, namespace=payload.namespace)


@router.get("/store/items/{key}", response_model=StoreItemResponse)
async def get_store_item(
    key: str,
    store: Annotated[BaseStore, Depends(get_memory_store)],
    space_id: Annotated[str, Query(min_length=1)],
    namespace: Annotated[str, Query(min_length=1)] = "default",
) -> StoreItemResponse:
    """Retrieve one persistent memory item isolated to `(space_id, namespace)`."""
    item = await store.aget(_space_namespace(space_id, namespace), key)
    if item is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Store item not found in space namespace",
        )
    return _to_item_response(item, space_id=space_id, namespace=namespace)


@router.post("/store/search", response_model=StoreSearchResponse)
async def search_store_items(
    payload: StoreSearchRequest,
    store: Annotated[BaseStore, Depends(get_memory_store)],
) -> StoreSearchResponse:
    """Query persistent memory items inside `(space_id, namespace)`."""
    results = await store.asearch(
        _space_namespace(payload.space_id, payload.namespace),
        query=payload.query,
        filter=payload.filter,
        limit=payload.limit,
    )
    return StoreSearchResponse(
        items=[
            _to_item_response(item, space_id=payload.space_id, namespace=payload.namespace)
            for item in results
        ]
    )


@router.delete("/store/items/{key}", response_model=StoreDeleteResponse)
async def delete_store_item(
    key: str,
    store: Annotated[BaseStore, Depends(get_memory_store)],
    space_id: Annotated[str, Query(min_length=1)],
    namespace: Annotated[str, Query(min_length=1)] = "default",
) -> StoreDeleteResponse:
    """Delete one persistent memory item from `(space_id, namespace)`."""
    await store.adelete(_space_namespace(space_id, namespace), key)
    return StoreDeleteResponse(space_id=space_id, namespace=namespace, key=key, deleted=True)


@router.delete("/spaces/{space_id}/store", response_model=SpaceStoreDeleteResponse)
async def delete_space_store(
    space_id: str,
    store: Annotated[BaseStore, Depends(get_memory_store)],
) -> SpaceStoreDeleteResponse:
    """Delete all long-term memory items across all namespaces of a Flux space."""
    namespaces = await store.alist_namespaces(prefix=("spaces", space_id))
    deleted_items = 0
    for ns in namespaces:
        while True:
            batch = await store.asearch(ns, limit=100)
            if not batch:
                break
            for item in batch:
                await store.adelete(ns, item.key)
                deleted_items += 1

    return SpaceStoreDeleteResponse(space_id=space_id, deleted_items=deleted_items)
