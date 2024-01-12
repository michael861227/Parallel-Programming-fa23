# Prerequisite for MPI

## 0. Initial Setting

```shell
$ mkdir -p ~/.ssh
$ ssh-keygen -t rsa # Leave all empty
```

## 1. Copy the config to `~/.ssh/config`

```Shell
$ cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
```

## 2. Enter pp2 to pp10

```Shell
ssh pp2
ssh pp3
.
.
.
ssh pp10
```

## 3. Maintain consistency by copying the data from the `.ssh` directory, ensuring that the keys on each computer are uniform

```shell
$ parallel-scp -A -h host.txt -r ~/.ssh ~
```

## 4. Ensure `SSH` access to another computer without requiring a password

## 5. Use parallel-scp to distribute the executable file to each server where you intend to execute it

For example

```shell
$ parallel-scp -h setting/hosts.txt -r ~/HW4 ~
```
