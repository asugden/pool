"""Script to sync CouchDB analysis database."""
import argparse

import couchdb

from pool.backends.couch_backend import Database

USER = 'admin'
PASSWORD = 'andermann'


def push(name='analysis', host='localhost', port=5984, remote_name='analysis',
         remote_host='tolman', remote_port=5984, user=None, password=None):
    """Push changes from one database to another.

    After pushing changes, all conflicting documents are removed from the
    database that was just pushed to.

    """
    print("Initiating push from {}@{} to {}@{}.".format(
        name, host, remote_name, remote_host))
    db = Database(name, host=host, port=port)
    remote_db = Database(
        remote_name, host=remote_host, port=remote_port)
    result = db.push(remote_db)

    if result.get('no_changes'):
        print("Push complete. No changes.")
    else:
        rep = _get_current_replication(result)
        print("Push complete. " +
              "{} changed doc(s) found, {} doc(s) written.".format(
                  rep.get('missing_found', 0), rep.get('docs_written', 0)))

    delete_conflicts(
        name=remote_name, host=remote_host, port=remote_port, user=user,
        password=password)

    return result


def pull(name='analysis', host='localhost', port=5984, remote_name='analysis',
         remote_host='tolman', remote_port=5984, user=None, password=None):
    """Pull changes in to one database from another.

    After pulling changes, all conflicting documents are removed from the
    database that was just pulled into.

    """
    print("Initiating pull from {}@{} to {}@{}.".format(
        remote_name, remote_host, name, host))
    db = Database(name, host=host, port=port, user=user, password=password)
    remote_db = Database(remote_name, host=remote_host, port=remote_port)
    result = db.pull(remote_db)

    if result.get('no_changes'):
        print("Pull complete. No changes.")
    else:
        rep = _get_current_replication(result)
        print("Pull complete. " +
              "{} changed doc(s) found, {} doc(s) written.".format(
                  rep.get('missing_found', 0), rep.get('docs_written', 0)))

    delete_conflicts(
        name=name, host=host, port=port, user=user, password=password)

    return result


def sync(name='analysis', host='localhost', port=5984, remote_name='analysis',
         remote_host='tolman', remote_port=5984, user=None, password=None):
    """Synchronize two databases.

    Sequentially does a push, pull, and compaction of both databases.

    """
    push(name=name, host=host, port=port, remote_name=remote_name,
         remote_host=remote_host, remote_port=remote_port)
    pull(name=name, host=host, port=port, remote_name=remote_name,
         remote_host=remote_host, remote_port=remote_port)
    compact(name=name, host=host, port=port, user=user, password=password)
    compact(name=remote_name, host=remote_host, port=remote_port, user=user,
            password=password)


def compact(
        name='analysis', host='localhost', port=5984, user=None,
        password=None):
    """Initiate compaction of database.

    Compaction removes all old versions of documents.

    """
    db = Database(name, host=host, port=port, user=user, password=password)
    result = db.compact()
    if result:
        print("Compaction of {}@{} successfully initiated.".format(name, host))
    else:
        print("!Compaction of {}@{} failed to initiate.".format(name, host))


def delete_conflicts(
        name='analysis', host='localhost', port=5984, user=None,
        password=None):
    """Identify and delete all documents in conflict.

    This will find conflicts and delete all versions of a document that has
    a conflict. Intended to trigger re-calculation of key/vals next time.

    """
    db = Database(name, host=host, port=port)

    conflicts = db.view('views/conflicts')
    try:
        _ = conflicts.total_rows
    except couchdb.http.ResourceNotFound:
        print("Adding conflict view.")
        _add_conflict_view(
            name=name, host=host, port=port, user=user, password=password)

    print("Beginning deletion of conflicting docs.")
    n_conflicts = 0
    for row in conflicts:
        n_conflicts += 1
        key = row['key']
        while key in db:
            db.delete(key)
    print("Deleted {} conflicting doc{} from {} on {}.".format(
        n_conflicts, '' if n_conflicts == 1 else 's', name, host))


def cleanup(
        name='analysis', host='localhost', port=5984, user=None,
        password=None):
    """Cleanup database by deleting conflicts and purging old documents."""
    delete_conflicts(
        name=name, host=host, port=port, user=user, password=password)
    compact(name=name, host=host, port=port, user=user, password=password)


def _add_conflict_view(
        name='analysis', host='localhost', port=5984, user=None,
        password=None):
    """Add a new view to the database to report conflicts.

    This should only ever need to be called once per database/server.

    """
    db = Database(name, host=host, port=port, user=user, password=password)
    _id = "_design/views"
    view = {
        "views": {
            "conflicts": {
                "map": "function(doc) {\n  if(doc._conflicts) {\n    emit(doc._id, doc);\n  }\n}"
            }
        },
        "language": "javascript"
    }
    db.put(_id=_id, **view)


def _get_current_replication(result):
    """Parse a replication result structure to pull out current replication."""
    return filter(
        lambda x: x['session_id'] == result['session_id'],
        result['history'])[0]


if __name__ == '__main__':
    argParser = argparse.ArgumentParser(description="""
        Helper script to sync multiple couchdb analysis databases. Currently
        supports the following commands: push, pull, sync, compact, and cleanup.
        """, epilog="""
        After every 'push' or 'pull' all conflicting documents are deleted.
        'sync' performs a push, then a pull, and then compacts both databases.
        """)
    argParser.add_argument(
        'command', action='store', type=str,
        help="database syncing command to run")
    argParser.add_argument(
        '-n', '--name', action="store", default='analysis',
        help="name of database to sync (default: analysis)")
    argParser.add_argument(
        '-N', '--remote_name', action="store", default='analysis',
        help="name of remote database to sync (default: analysis)")
    argParser.add_argument(
        '-r', '--remote_host', action="store", default='tolman',
        help='hostname of remote server (default: tolman)')
    args = argParser.parse_args()

    if args.command == 'push':
        push(name=args.name, remote_name=args.remote_name,
             remote_host=args.remote_host, user=USER, password=PASSWORD)
    elif args.command == 'pull':
        pull(name=args.name, remote_name=args.remote_name,
             remote_host=args.remote_host, user=USER, password=PASSWORD)
    elif args.command == 'sync':
        sync(name=args.name, remote_name=args.remote_name,
             remote_host=args.remote_host, user=USER, password=PASSWORD)
    elif args.command == 'compact':
        compact(name=args.name, user=USER, password=PASSWORD)
    elif args.command == 'cleanup':
        cleanup(name=args.name, user=USER, password=PASSWORD)
