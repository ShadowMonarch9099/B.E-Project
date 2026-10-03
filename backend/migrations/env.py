from alembic import context

from app.db.session import get_engine

target_metadata = None  # migrations are written by hand


def run_migrations_online() -> None:
    with get_engine().connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata,
                          render_as_batch=connection.dialect.name == "sqlite")
        with context.begin_transaction():
            context.run_migrations()
        connection.commit()


run_migrations_online()
