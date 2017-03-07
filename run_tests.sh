#!/bin/bash

nosetests test/ 2>&1 >/dev/null | grep -v '^tensorflow: Level' | grep -v '^Level [0-9]:tensorflow' | egrep -v 'E?I tensorflow' | grep -v '^E$' | egrep -v '(begin|end) captured logging'
